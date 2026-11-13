import os
import numpy as np
import torch
import pandas as pd
import torchattacks
from tab_ddpm import load_pretrained_classifier
from sklearn.preprocessing import StandardScaler

# 參數設定
attack_method = 'cw'  # 選擇 'fgsm'、'pgd' 或 'cw'
epsilon = 0.1           # FGSM/PGD強度
pgd_alpha = 0.01        # PGD 單步強度
pgd_steps = 40          # PGD 步數

# 載入資料
# original
X_num = torch.tensor(np.load('data/colosseum/X_num_train.npy', allow_pickle=True), dtype=torch.float32)
X_cat_data = np.load('data/colosseum/X_cat_train.npy', allow_pickle=True)
X_cat_data_fixed = X_cat_data.astype(np.int64)
X_cat = torch.tensor(np.array(X_cat_data_fixed, dtype=np.int64), dtype=torch.long)
y_true = torch.tensor(np.load('data/colosseum/y_train.npy', allow_pickle=True), dtype=torch.long)

# for original
scaler = StandardScaler()
X_num_np = X_num.cpu().numpy()
X_num_np = scaler.fit_transform(X_num_np)
X_num_scaled = torch.tensor(X_num_np, dtype=torch.float32, device=X_num.device)

# 載入模型
classifier = load_pretrained_classifier()
device = next(classifier.parameters()).device

# 整理資料
X = torch.cat([X_num, X_cat], dim=1)
slice_id = X[:, -2].to(device) # slice_id位置（假設倒數第二個欄位）
X = X[:, torch.arange(X.shape[1]) != X.shape[1] - 2]  # remove slice_id
X = X.unsqueeze(1)
X, y_true = X.to(device), y_true.to(device)

# 選擇攻擊方法
if attack_method == 'fgsm':
    attack = torchattacks.FGSM(classifier, eps=epsilon)
elif attack_method == 'pgd':
    attack = torchattacks.PGD(classifier, eps=epsilon, alpha=pgd_alpha, steps=pgd_steps)
elif attack_method == 'cw':
    attack = torchattacks.CW(classifier, c=1.0, kappa=0, steps=100)
else:
    raise ValueError("Unsupported attack method")

# 產生對抗樣本
classifier.eval()
with torch.no_grad():
    pred_logits = classifier(X)

pred_y = pred_logits.argmax(dim=1)
true_logit = pred_logits[torch.arange(pred_logits.size(0)), y_true]
logits_clone = pred_logits.clone()
logits_clone[torch.arange(pred_logits.size(0)), y_true] = float('-inf')
max_other_logit, _ = logits_clone.max(dim=1)
confidence_diff_logit = true_logit - max_other_logit

# 執行攻擊
X_adv = attack(X, y_true)
X_adv = X_adv.squeeze(1)  # 還原形狀

# 還原數值和類別特徵
X_num_adv = X_adv[:, :X_num.shape[1]]
X_cat_adv = X_adv[:, X_num.shape[1]:]

# slice_id 需插入對抗後的 X_cat_adv
slice_id_tensor = slice_id.clone().detach().unsqueeze(1)  # (N, 1)
# 原本 X_cat_adv 有兩個欄位：(a, b)
# 拆分前半與後半
X_cat_first = X_cat_adv[:, :1]   # 第一個欄位 a
X_cat_second = X_cat_adv[:, 1:]  # 第二個欄位 b
# 插入 slice_id，形成三欄：(a, slice_id, b)
X_cat_adv = torch.cat([X_cat_first, slice_id_tensor, X_cat_second], dim=1)

# 確保資料移回CPU存檔
X_num_np = X_num.cpu().numpy()
X_cat_np = X_cat.cpu().numpy()
X_num_adv_np = X_num_adv.cpu().numpy()
X_cat_adv_np = X_cat_adv.cpu().numpy()
y_true_np = y_true.cpu().numpy()
pred_y_np = pred_y.cpu().numpy()
pred_logits_np = pred_logits.cpu().numpy()

# 儲存對抗資料
os.makedirs('data/colosseum/adversarial_examples', exist_ok=True)
np.save('data/colosseum/adversarial_examples/X_num_train.npy', X_num_adv_np)
np.save('data/colosseum/adversarial_examples/X_cat_train.npy', X_cat_adv_np)
np.save('data/colosseum/adversarial_examples/y_train.npy', y_true_np)

# 生成CSV紀錄
records = []
for i in range(len(y_true_np)):
    record = {
        'true_label': y_true_np[i],
        'predicted_label': pred_y_np[i],
    }
    # 儲存每個類別的logit
    for cls in range(pred_logits_np.shape[1]):
        record[f'logit_class_{cls}'] = pred_logits_np[i, cls]

    # 儲存原始特徵
    for j in range(X_num_np.shape[1]):
        record[f'orig_X_num_{j}'] = X_num_np[i, j]
    for j in range(X_cat_np.shape[1]):
        record[f'orig_X_cat_{j}'] = X_cat_np[i, j]

    # 儲存對抗特徵
    for j in range(X_num_adv_np.shape[1]):
        record[f'adv_X_num_{j}'] = X_num_adv_np[i, j]
    for j in range(X_cat_adv_np.shape[1]):
        record[f'adv_X_cat_{j}'] = X_cat_adv_np[i, j]

    records.append(record)

df = pd.DataFrame(records)
df.to_csv('data/colosseum/adversarial_examples/adversarial_report.csv', index=False)

print(f"Attack complete, data already store to data/colosseum/adversarial_examples/")
