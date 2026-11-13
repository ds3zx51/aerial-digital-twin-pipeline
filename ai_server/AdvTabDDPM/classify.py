import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tab_ddpm import load_pretrained_classifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # add for npy classify
import os

# 載入 combine 資料
X_num = torch.tensor(np.load('data/colosseum/combine/X_num_train.npy'), dtype=torch.float32)
X_cat = torch.tensor(np.load('data/colosseum/combine/X_cat_train.npy'), dtype=torch.long)
y_true = torch.tensor(np.load('data/colosseum/combine/y_train.npy'), dtype=torch.long)

# # 載入 combine perturbed test 資料
# X_num = torch.tensor(np.load('data/colosseum/combine/1_1/X_num_test.npy'), dtype=torch.float32)
# X_cat = torch.tensor(np.load('data/colosseum/combine/1_1/X_cat_test.npy'), dtype=torch.long)
# y_true = torch.tensor(np.load('data/colosseum/combine/1_1/y_test.npy'), dtype=torch.long)

# # 載入 adversarial attack 資料
# X_num = torch.tensor(np.load('data/colosseum/adversarial_examples/CW/X_num_train.npy'), dtype=torch.float32)
# X_cat = torch.tensor(np.load('data/colosseum/adversarial_examples/CW/X_cat_train.npy'), dtype=torch.long)
# y_true = torch.tensor(np.load('data/colosseum/adversarial_examples/CW/y_train.npy'), dtype=torch.long)

# # 載入 perturbed 資料
# X_num = torch.tensor(np.load('exp/colosseum/ddpm_tune_best/X_num_train.npy'), dtype=torch.float32)
# X_cat = torch.tensor(np.load('exp/colosseum/ddpm_tune_best/X_cat_train.npy'), dtype=torch.long)
# y_true = torch.tensor(np.load('exp/colosseum/ddpm_tune_best/y_train.npy'), dtype=torch.long)
# # y_true = torch.tensor(np.load('exp/colosseum/ddpm_tune_best/y_uniform_neg.npy'), dtype=torch.long)

# # original
# X_num = torch.tensor(np.load('data/colosseum/X_num_train.npy', allow_pickle=True), dtype=torch.float32)
# # X_cat = torch.tensor(np.load('exp/colosseum/ddpm_tune_best/X_cat_train.npy', allow_pickle=True), dtype=torch.long)
# X_cat_data = np.load('data/colosseum/X_cat_train.npy', allow_pickle=True)
# X_cat_data_fixed = X_cat_data.astype(np.int64)
# X_cat = torch.tensor(np.array(X_cat_data_fixed, dtype=np.int64), dtype=torch.long)
# y_true = torch.tensor(np.load('data/colosseum/y_train.npy', allow_pickle=True), dtype=torch.long)

# # synthetic
# X_num = torch.tensor(np.load('data/colosseum/before_adversarial/X_num_train.npy', allow_pickle=True), dtype=torch.float32)
# # X_cat = torch.tensor(np.load('exp/colosseum/ddpm_tune_best/X_cat_train.npy', allow_pickle=True), dtype=torch.long)
# X_cat_data = np.load('data/colosseum/before_adversarial/X_cat_train.npy', allow_pickle=True)
# X_cat_data_fixed = X_cat_data.astype(np.int64)
# X_cat = torch.tensor(np.array(X_cat_data_fixed, dtype=np.int64), dtype=torch.long)
# y_true = torch.tensor(np.load('data/colosseum/before_adversarial/y_train.npy', allow_pickle=True), dtype=torch.long)

classifier = load_pretrained_classifier()
# 移到相同裝置（假設 classifier 在 GPU）
device = next(classifier.parameters()).device

# # for original and synthetic and adversarial attack
# scaler = StandardScaler()
# X_num_np = X_num.cpu().numpy()
# X_num_np = scaler.fit_transform(X_num_np)
# X_num_scaled = torch.tensor(X_num_np, dtype=torch.float32, device=X_num.device)
# X = torch.cat([X_num_scaled, X_cat], dim=1)

X = torch.cat([X_num, X_cat], dim=1)

X = X[:, torch.arange(X.shape[1]) != X.shape[1] - 2]  # remove slice_id
X = X.unsqueeze(1)
X, y_true = X.to(device), y_true.to(device)

# 取得 logits
with torch.no_grad():
    pred_logits = classifier(X)  # shape: (N, num_classes)

# 計算 confidence_diff_logit
true_logit = pred_logits[torch.arange(pred_logits.size(0)), y_true]
logits_clone = pred_logits.clone()
logits_clone[torch.arange(pred_logits.size(0)), y_true] = float('-inf')
max_other_logit, _ = logits_clone.max(dim=1)
confidence_diff_logit = true_logit - max_other_logit

# 判別是否在 delta 範圍內
delta = 0.93#1.71#1.0
is_within_delta = (confidence_diff_logit.abs() < delta)

# 判別類別
y_pred = torch.argmax(pred_logits, dim=1)

# 統計 confusion matrix
y_true_np = y_true.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()

print("Confusion Matrix:")
print(confusion_matrix(y_true_np, y_pred_np))
print()

# 顯示分類報告（精確度、召回率、F1）
print("Classification Report:")
print(classification_report(y_true_np, y_pred_np, target_names=["eMBB", "mMTC", "URLLC"]))

# 額外顯示 confidence_diff_logit 統計
print("Confidence diff satistics")
print(f"Mean: {confidence_diff_logit.mean().item():.4f}, Std: {confidence_diff_logit.std().item():.4f}")
print(f"Percentage |confidence_diff_logit| < delta ({delta}): {(is_within_delta.float().mean().item() * 100):.2f}%")

# --- 混淆矩陣 ---
cm = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy(), normalize='true')
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=["eMBB", "mMTC", "URLLC"], yticklabels=["eMBB", "mMTC", "URLLC"], fmt=".2f")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# --- Confidence diff 分佈 ---
# 全域字體設定
plt.rcParams.update({
    'font.size': 17,       # 全域字體大小
    'axes.titlesize': 18,  # 標題字體
    'axes.labelsize': 18,  # 座標軸標籤字體
    'xtick.labelsize': 16, # x軸刻度字體
    'ytick.labelsize': 16, # y軸刻度字體
    'legend.fontsize': 15  # 圖例字體
})
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
plt.figure(figsize=(8, 5))
for cls in range(3):
    mask = y_true == cls
    plt.hist(confidence_diff_logit[mask].cpu().numpy(), bins=100, alpha=0.6, label=["eMBB", "mMTC", "URLLC"][cls], color=colors[cls])
# plt.axvline(x=-delta, color='black', linestyle='--', linewidth=1)
# plt.axvline(x=delta, color='black', linestyle='--', linewidth=1)
offset = 0.05  # 可以視你的 delta 大小調整
# 左邊 -delta
plt.axvline(x=-delta, color='black', linestyle='--', linewidth=1)
plt.text(-delta - offset, plt.ylim()[1]*0.9, f"-δ = {-delta:.2f}", rotation=90,
         verticalalignment='top', horizontalalignment='right', color='black')
# 右邊 +delta
plt.axvline(x=delta, color='black', linestyle='--', linewidth=1)
plt.text(delta + offset, plt.ylim()[1]*0.9, f"δ = {delta:.2f}", rotation=90,
         verticalalignment='top', horizontalalignment='left', color='black')
plt.xlabel("Confidence Difference (logit)")
plt.ylabel("Number of Samples")
# plt.title("Distribution of Confidence Difference by Class")
plt.legend()
plt.tight_layout()
plt.savefig("confidence_diff_distribution.png")
plt.close()

# --- 每類別的準確率 (Accuracy on clean data) ---
print("Per-class Accuracy on Clean Data:")
for cls in range(3):
    cls_mask = (y_true == cls)
    cls_acc = (y_pred[cls_mask] == y_true[cls_mask]).float().mean().item()
    print(f"{['eMBB', 'mMTC', 'URLLC'][cls]}: {cls_acc * 100:.2f}%")

# --- 總準確率 ---
total_acc = (y_pred == y_true).float().mean().item()
print(f"\nTotal Accuracy on Clean Data: {total_acc * 100:.2f}%")

# # --- 每類別的魯棒性 (Accuracy on adversarial data) ---
# print("\nPer-class Robustness (Accuracy on Adversarial Samples):")
# for cls in range(3):
#     cls_mask = (y_true == cls)
#     cls_robust = (y_pred_adv[cls_mask] == y_true[cls_mask]).float().mean().item()
#     print(f"{['eMBB', 'mMTC', 'URLLC'][cls]}: {cls_robust * 100:.2f}%")

# # --- 總魯棒性 ---
# total_robust = (y_pred_adv == y_true).float().mean().item()
# print(f"\nTotal Robustness (Adversarial Accuracy): {total_robust * 100:.2f}%")

import pandas as pd
# 將結果轉為 numpy
y_true_np = y_true.cpu().numpy()
y_pred_np = y_pred.cpu().numpy()
logits_np = pred_logits.cpu().numpy()
confidence_diff_np = confidence_diff_logit.cpu().numpy()

# 儲存 y_true、y_pred、各類別 logit 到 CSV
df = pd.DataFrame({
    "index": np.arange(len(y_true_np)),
    "y_true": y_true_np,
    "y_pred": y_pred_np,
    "logit_eMBB": logits_np[:, 0],
    "logit_mMTC": logits_np[:, 1],
    "logit_URLLC": logits_np[:, 2],
    "confidence_diff_logit": confidence_diff_np
})
df.to_csv("results.csv", index=False)
print(f"\n[INFO] 已儲存 results.csv, 總共 {len(df)} 筆資料。")

# 計算分類正確與錯誤比例
correct_mask = (y_true_np == y_pred_np)
num_correct = correct_mask.sum()
num_total = len(y_true_np)
acc_ratio = num_correct / num_total * 100
error_ratio = 100 - acc_ratio

print(f"\n分類正確比例: {acc_ratio:.2f}%")
print(f"分類錯誤比例: {error_ratio:.2f}%")

# 計算分類錯誤的 confidence_diff 平均值
error_conf_diff = confidence_diff_np[~correct_mask]
if len(error_conf_diff) > 0:
    mean_error_conf_diff = error_conf_diff.mean()
    print(f"分類錯誤樣本的 confidence difference 平均值: {mean_error_conf_diff:.4f}")
else:
    print("所有樣本皆分類正確，無錯誤樣本的 confidence difference 統計")

# 儲存分類錯誤的 index 與 confidence difference
mis_idx = np.where(~correct_mask)[0]
mis_conf_diff = confidence_diff_np[~correct_mask]

df_mis = pd.DataFrame({
    "index": mis_idx,
    "confidence_diff_logit": mis_conf_diff
})
df_mis.to_csv("misclassified_details.csv", index=False)
print(f"[INFO] 已儲存 misclassified_details.csv, 分類錯誤樣本數: {len(df_mis)}")


# uncertainty = True  # 控制是否啟用 Uncertainty Injection
uncertainty = False
alpha = 0  # 推測類別的權重，可依需求調整
save_dir = "exp/colosseum/ddpm_tune_best"
os.makedirs(save_dir, exist_ok=True)


# 2. 標準化 confidence difference (可針對 boundary_mask 區域)
boundary_mask = is_within_delta
cd = confidence_diff_logit[boundary_mask]
cd_min = cd.min()
cd_max = cd.max()
eps = 1e-6
cd_norm = (cd - cd_min) / (cd_max - cd_min + eps)  # 越接近邊界 → 越接近 0
# 3. 設定 alpha 動態調整範圍
min_alpha = 0.0  # 越小不確定性越大
max_alpha = 0.8  # 保留原始預測
# 4. 動態 alpha 設定
alpha_dynamic_boundary = min_alpha + (max_alpha - min_alpha) ** cd_norm


num_classes = pred_logits.shape[1]
if uncertainty:
    with torch.no_grad():
        prob_logits = classifier(X)
        prob = torch.softmax(prob_logits, dim=1)  # (N, num_classes)

    uniform = torch.full_like(prob, 1.0 / num_classes)

    y_pred = torch.argmax(prob, dim=1)
    boundary_mask = is_within_delta
    neg_mask = boundary_mask & (y_pred != y_true)  # 僅錯誤分類區域

    # # 動態 alpha，可根據樣本信心決定，這裡示範固定
    # alpha_dynamic = alpha
    # temperature = 2.0  # T > 1 降低信心
    # prob_scaled = torch.softmax(prob_logits / temperature, dim=1)
    # y_mix = alpha_dynamic * prob_scaled + (1 - alpha_dynamic) * uniform

    # 1. 排序機率，找出前兩名
    top2_probs, top2_indices = prob.topk(2, dim=1)  # (N, 2)
    # 2. 真實標籤
    y_true = y_true.to(prob.device)
    # 3. 構建新標籤，若 top1 等於真實標籤，選 top2
    # 否則保留原標籤（也可視需求直接換 top2，不考慮真實標籤）
    y_mix_label = top2_indices[:, 1]  # 取第二高機率類別

    # # 預設全域 alpha
    # temperature = 2.0  # 溫度越大，機率越平坦
    # prob_scaled = torch.softmax(prob_logits / temperature, dim=1)
    # alpha_dynamic = torch.full((X.shape[0],), fill_value=max_alpha, device=X.device)
    # alpha_dynamic[boundary_mask] = alpha_dynamic_boundary
    # # 6. 最終 label 混合
    # y_mix = alpha_dynamic.unsqueeze(1) * prob_scaled + (1 - alpha_dynamic.unsqueeze(1)) * uniform
    
    # y_mix_label = torch.argmax(y_mix, dim=1)

    y_full_uniform_neg = y_true.clone()
    changed_mask = neg_mask & (y_mix_label != y_pred)
    y_full_uniform_neg[changed_mask] = y_mix_label[changed_mask]

    np.save(os.path.join(save_dir, "y_uniform_neg.npy"), y_full_uniform_neg.cpu().numpy())
    print(f"Dynamic UEI done. Total {changed_mask.sum().item()} samples modified.")

    ### 類別轉換與正確/錯誤分類統計 ###
    changed_from = y_pred[changed_mask]
    changed_to = y_mix_label[changed_mask]
    changed_true = y_true[changed_mask]  # 原本模型預測

    num_classes = prob.shape[1]
    transition_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    correct_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)
    wrong_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int32)

    for from_label, to_label, true_label in zip(changed_from, changed_to, changed_true):
        transition_matrix[from_label, to_label] += 1

        if true_label == to_label:
            correct_matrix[from_label, to_label] += 1  # 轉換後預測正確
        else:
            wrong_matrix[from_label, to_label] += 1    # 轉換後預測錯誤

    print("\nclass transform statistics (from → to): ")
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and transition_matrix[i, j] > 0:
                total = transition_matrix[i, j].item()
                correct = correct_matrix[i, j].item()
                wrong = wrong_matrix[i, j].item()
                print(f"class {i} → class {j} : {total} samples, correct {correct} samples, incorrect {wrong} samples")

    # 只針對 changed_mask 的樣本統計
    idx_selected = changed_mask.nonzero(as_tuple=True)[0]

    # 準備資料
    data_list = []
    for idx in idx_selected.cpu().numpy():
        row = {
            "index": idx,  # 新增 index 欄位
            "true_label": y_true[idx].item(),
            "pred_label": y_pred[idx].item(),
            "second_label": y_mix_label[idx].item(),
        }
        # 各類別原始 prob
        for cls in range(num_classes):
            row[f"prob_cls{cls}"] = prob[idx, cls].item()

        # # 經過 label smoothing 混合的 label
        # for cls in range(num_classes):
        #     row[f"y_mix_cls{cls}"] = y_mix[idx, cls].item()

        data_list.append(row)

    # 轉成 DataFrame
    df = pd.DataFrame(data_list)

    # 儲存
    save_path = os.path.join(save_dir, "UEI_label_statistics.csv")
    df.to_csv(save_path, index=False)

    print(f"Already save detail information to {save_path}")

# if uncertainty:
#     with torch.no_grad():
#         prob_logits = classifier(X)
#         prob = torch.softmax(prob_logits, dim=1)  # (N, num_classes)

#     uniform = torch.full_like(prob, 1.0 / num_classes)
#     y_mix = alpha * prob + (1 - alpha) * uniform
#     y_mix_label = torch.argmax(y_mix, dim=1)

#     boundary_mask = is_within_delta

#     ### (1) 儲存完整 y ，僅 boundary 內有變化的修改 ###
#     y_full_uniform = y_true.clone()
#     changed_mask = boundary_mask & (y_mix_label != y_true)
#     y_full_uniform[changed_mask] = y_mix_label[changed_mask]
#     np.save(os.path.join(save_dir, "y_uniform_full.npy"), y_full_uniform.cpu().numpy())
#     print(f"Already store Uniform Injection label to {save_dir}/y_uniform_full.npy")
#     print(f"Total {changed_mask.sum().item()} samples label be modified.")

#     ### (2) 錯誤分類區域的完整對應 ###
#     neg_mask = boundary_mask & (y_pred != y_true)
#     y_full_uniform_neg = y_true.clone()
#     neg_changed_mask = neg_mask & (y_mix_label != y_true)
#     y_full_uniform_neg[neg_changed_mask] = y_mix_label[neg_changed_mask]

#     np.save(os.path.join(save_dir, "y_uniform_neg.npy"), y_full_uniform_neg.cpu().numpy())
#     print(f"Already store boundary zone misclassification Uniform Injection label to {save_dir}/y_uniform_neg.npy")
#     print(f"Total {neg_changed_mask.sum().item()} samples label be modified.")
