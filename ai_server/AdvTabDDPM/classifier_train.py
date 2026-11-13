import argparse
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import pickle
from torch import nn
from torch.utils.data import DataLoader
import tddpm_lib

from sklearn.metrics import confusion_matrix as conf_mat
from sklearn.metrics import f1_score
import seaborn as sn

import os
proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
import sys
sys.path.append(proj_root_dir)
import pandas as pd
import classifier_models
from classifier_models import ConvNN, TransformerNN, TransformerNN_v2
from classifier_models import megatron_ViT as ViT
from classifier_dataset import TracesDataset
from classifier_data_preprocess import safe_pickle_dump, gen_slice_dataset_from_numpy #npy_to_pkl

def train_epoch(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    model.train()
    start_time = time.time()

    y_pred_all = []
    y_true_all = []
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler_fitted = False
    delta = 0.93 # add for label smoothing
    min_scale = 0.1  # add for boundary zone label smoothing # 最小損失比例，完全不信任樣本
    max_scale = 1.0  # add for boundary zone label smoothing # 高置信度樣本不改變
    for batch, (X_num, X_cat, y) in enumerate(dataloader):
        X = torch.cat([X_num, X_cat], dim=1)
        X = X[:, torch.arange(X.shape[1]) != X.shape[1] - 2]  # remove slice_id
        X = X.unsqueeze(1)


        X = X.to(device)
        y = y.to(device)
        
        pred = model(X)

        '''
        # add for label smoothing
        true_logit = pred[torch.arange(pred.size(0)), y]
        logits_clone = pred.clone()
        logits_clone[torch.arange(pred.size(0)), y] = float('-inf')
        max_other_logit, _ = logits_clone.max(dim=1)
        confidence_diff_logit = (true_logit - max_other_logit).detach()

        # add for boundary zone label smoothing
        # 動態 loss scale 計算：低置信度 → scale 趨近 min_scale，高置信度 → scale = max_scale
        soft_label_scale = torch.ones_like(confidence_diff_logit, device=device) * max_scale
        mask_low_conf = confidence_diff_logit < delta
        scale_low_conf = (confidence_diff_logit[mask_low_conf] / delta).clamp(0, 1)
        scale_low_conf = min_scale + (max_scale - min_scale) * scale_low_conf  # 線性映射
        soft_label_scale[mask_low_conf] = scale_low_conf
        # 傳入 FocalLoss
        loss = loss_fn(pred, y, soft_label_scale=soft_label_scale)


        # # 動態 λ(δ) 控制不確定性比例
        # lambda_vals = torch.clamp(confidence_diff_logit / delta, min=0.0, max=1.0) # modify for boundary zone label smoothing
        # # 傳入 FocalLoss
        # loss = loss_fn(pred, y, soft_label_scale=lambda_vals) # modify for boundary zone label smoothing
        '''

        loss = loss_fn(pred, y) # modify for label smoothing

        prob = torch.argmax(pred, dim=1)
        probs = torch.softmax(pred, dim=1)
        y_pred_all.extend(prob.cpu().tolist())
        y_true_all.extend(y.cpu().tolist())

        # print(f'X {X}, y_pred {prob}, pred {pred}, y {y}, correct {sum(prob == y)/len(y) * 100}, loss {loss}')

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping # add
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(classification_report(y_true_all, y_pred_all))
            # print("Logits (before softmax):", pred[:5], "Softmax:", probs[:5], "Logit diffs:", (pred.max(dim=1)[0] - pred.min(dim=1)[0]).mean())
            # print("Max Prob:", probs.max(dim=1)[0].mean().item(), "Class:", torch.argmax(pred, dim=1).unique())
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print("--- %s seconds ---" % (time.time() - start_time))
    return (time.time() - start_time)

def validate_epoch(dataloader, model, loss_fn, Nclasses, logdir='.', device="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    conf_matrix = np.zeros((Nclasses, Nclasses))
    
    # add for npy classify
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler_fitted = False

    with torch.no_grad():
        for X_num, X_cat, y in dataloader:
        # for X, y in dataloader: # modify for npy classifier
            # add for npy classifier
            if not scaler_fitted:
                X_num_np = X_num.cpu().numpy()
                X_num_np = scaler.fit_transform(X_num_np)
                scaler_fitted = True
            else:
                X_num_np = X_num.cpu().numpy()
                X_num_np = scaler.transform(X_num_np)

            X_num_scaled = torch.tensor(X_num_np, dtype=torch.float32, device=X_num.device)
            
            X = torch.cat([X_num_scaled, X_cat], dim=1)
            X = X[:, torch.arange(X.shape[1]) != X.shape[1] - 2]  # remove slice_id
            X = X.unsqueeze(1)


            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            conf_matrix += conf_mat(y.cpu(), pred.argmax(1).cpu(), labels=list(range(Nclasses)))
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.1f}%, "
        f"Avg loss: {test_loss:>8f} \n"
        f"conf_matrix: {conf_matrix}"
    )

    return test_loss, conf_matrix

# add for balance loss
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target, soft_label_scale=None):
        """
        input: logits [B, C]
        target: 真實類別 [B]
        soft_label_scale: 每個樣本的 λ(δ) 係數，用來平衡 one-hot 與均勻分佈
        """
        logp = F.log_softmax(input, dim=1) # logits -> log-probs
        p = torch.exp(logp)

        target_logp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        target_p = p.gather(1, target.unsqueeze(1)).squeeze(1)

        focal_term = (1 - target_p) ** self.gamma
        loss = -focal_term * target_logp

        if soft_label_scale is not None:
            # 針對不確定樣本，降低 one-hot 標籤影響
            # λ(δ) → 越小越不確定，損失降低，防止強制擬合
            loss = loss * soft_label_scale

        if self.weight is not None:
            weight = self.weight[target] if self.weight.ndim == 1 else self.weight
            loss = loss * weight.to(input.device)

        return loss.mean()

# add for good result
def diagonal_accuracy(cm):
    correct = np.trace(cm)  # 對角線的和 = 分類正確的數量
    total = np.sum(cm)      # 所有樣本數
    return correct / total

# add for npy classify
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset
def train_func(config: Dict, ds_train=None, ds_test=None):
    batch_size = config["batch_size"]
    lr = config["lr"]
    lrmin = config["lrmin"]
    lrpatience = config["lrpatience"]
    epochs = config["epochs"]
    Nclass = config["Nclass"]
    slice_len = config['slice_len']
    num_feats = config['num_feats']
    global_model = config['global_model']
    model_postfix = config['model_postfix']
    device = config['device']
    logdir = config['logdir']
    pos_enc = config['pos_enc']
    patience = config['patience']

    data_dir = 'data/colosseum_22'
    # data_dir = 'data/colosseum'
    # === 載入 .npy 資料 ===
    X_num_train = torch.tensor(np.load(os.path.join(data_dir, "X_num_train.npy")), dtype=torch.float32)
    X_cat_train = torch.tensor(np.array(np.load(os.path.join(data_dir, "X_cat_train.npy"), allow_pickle=True).astype(np.int64), dtype=np.int64), dtype=torch.long)
    y_train = torch.tensor(np.load(os.path.join(data_dir, "y_train.npy"), allow_pickle=True).astype(np.int64), dtype=torch.long)

    X_num_test = torch.tensor(np.load(os.path.join(data_dir, "X_num_test.npy")), dtype=torch.float32)
    X_cat_test = torch.tensor(np.array(np.load(os.path.join(data_dir, "X_cat_test.npy"), allow_pickle=True).astype(np.int64), dtype=np.int64), dtype=torch.long)
    y_test = torch.tensor(np.load(os.path.join(data_dir, "y_test.npy")), dtype=torch.long)

    # 確保長度一致
    assert len(X_num_train) == len(X_cat_train) == len(y_train)

    # === 建立 Dataset 與 DataLoader ===
    dataset_train = TensorDataset(X_num_train, X_cat_train, y_train)
    dataset_test = TensorDataset(X_num_test, X_cat_test, y_test)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)  # 你也可以分割 test

    # === 建立 Model 與 Loss ===
    model, loss_fn = TRACTOR_model(Nclass, global_model, num_feats, slice_len, pos_enc, dropout=config['dropout'])
    model.to(device)

    # === 平衡 Loss 設定 ===
    use_focal = True
    focal_gamma = 2.0
    use_class_weights = True
    class_weights = None

    if use_class_weights:
        y_train_all = y_train.cpu().numpy()
        class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train_all), y=y_train_all)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
        print('class_weights: ', class_weights)

    if use_focal:
        loss_fn = FocalLoss(gamma=focal_gamma, weight=class_weights)
        # loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.NLLLoss(weight=class_weights)

    # === Optimizer & Scheduler ===
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=lrpatience, min_lr=lrmin)

    # === Logging ===
    model_name = f'model.{slice_len}.{model_postfix}.pt'
    best_loss = np.inf
    best_diag_acc = 0.0
    epochs_wo_improvement = 0
    times = []
    last_lr = np.inf
    loss_results = []

    # === Model Summary ===
    print(model)
    for name, param in model.named_parameters():
        print(f'{name:20} {param.numel()} {list(param.shape)}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'TOTAL                {total_params}')

    # === Training Loop ===
    for e in range(epochs):
        ep_time = train_epoch(train_dataloader, model, loss_fn, optimizer, device)
        times.append(ep_time)
        loss, cm = validate_epoch(test_dataloader, model, loss_fn, Nclasses=Nclass, device=device)

        scheduler.step(loss)
        if scheduler._last_lr[0] < last_lr:
            if last_lr == np.inf:
                last_lr = scheduler._last_lr[0]
            else:
                print("[lr change detect] -> Reloading from best state, skipping epoch")
                last_lr = scheduler._last_lr[0]
                model.load_state_dict(torch.load(os.path.join(logdir, model_name), map_location=device)['model_state_dict'])
                continue

        loss_results.append(loss)
        epochs_wo_improvement += 1
        diag_acc = diagonal_accuracy(cm)

        if diag_acc > best_diag_acc:
            print('best_loss > loss is: ', best_loss > loss)
            print('(np.isclose(best_loss, loss) and diag_acc > best_diag_acc) is: ', diag_acc > best_diag_acc)
            pickle.dump(cm, open(os.path.join(logdir, 'conf_matrix.last.pkl'), 'wb'))
            epochs_wo_improvement = 0
            best_loss = loss
            best_diag_acc = diag_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(logdir, model_name))

        if epochs_wo_improvement > patience:
            print('------------------------------------')
            print('Early termination implemented at epoch:', e+1)
            print('------------------------------------')
            print(f'Training time analysis for model {model_postfix} with slice {slice_len}:')
            sd = np.std(times)
            mean = np.mean(times)
            print(f'Mean: {mean}, std: {sd}')
            timing_file = os.path.join(logdir, 'tr_time.txt')
            with open(timing_file, 'a') as f:
                f.write(f'Model {model_postfix} with slice {slice_len}:\n')
                f.write(f'Mean: {mean}, std: {sd}, num. epochs: {e+1}\n')
            return loss_results, model_name

    return loss_results, model_name

def convert_npy_to_pickle(data_dir, use_diffusion_data=False, output_dir='data/pickle', slice_len=1, dataset_filename='diffusion_dataset_globalnorm.pkl', norm_filename='diffusion_global_cols_maxmin.pkl'):
    print("Loading diffusion-generated evaluation data...")

    dataset, cols_maxmin = gen_slice_dataset_from_numpy(data_dir, use_diffusion_data=use_diffusion_data, slice_len=slice_len)

    os.makedirs(output_dir, exist_ok=True)
    if isinstance(dataset_filename, list):
        dataset_filename = dataset_filename[0]
    if isinstance(norm_filename, list):
        norm_filename = norm_filename[0]
    dataset_path = os.path.join(output_dir, dataset_filename)
    norm_param_path = os.path.join(output_dir, norm_filename)

    # Save raw pickle
    safe_pickle_dump(dataset_path, dataset)
    safe_pickle_dump(norm_param_path, cols_maxmin)

    return dataset_filename, norm_filename, dataset, cols_maxmin

# add for CD
def save_cd_distribution_plot(confidence_differences, delta, parent_dir):
    plt.figure(figsize=(8, 6))
    plt.hist(confidence_differences, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(x=delta, color='red', linestyle='--', label=f'+δ={delta}')
    plt.axvline(x=-delta, color='blue', linestyle='--', label=f'-δ={-delta}')
    plt.title('Confidence Difference (CD) Distribution')
    plt.xlabel('Confidence Difference (CD)')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    cd_plot_path = os.path.join(parent_dir, 'confidence_difference_distribution.png')
    plt.savefig(cd_plot_path)
    plt.close()
    print(f"CD distribution plot saved to {cd_plot_path}")

def find_best_delta(conf_diffs, labels, preds, delta_candidates=None, alpha=0.5, beta=0.5):
    if delta_candidates is None:
        delta_candidates = np.linspace(0.01, 0.5, 50)  # 掃描範圍

    best_score = -np.inf
    best_delta = 0.2  # default
    conf_diffs = np.array(conf_diffs)
    labels = np.array(labels)
    preds = np.array(preds)

    for delta in delta_candidates:
        boundary_flags = (np.abs(conf_diffs) <= delta)
        general_flags = ~boundary_flags

        N_total = len(labels)
        N_boundary = boundary_flags.sum()
        N_general = general_flags.sum()

        # Boundary zone 比例
        boundary_ratio = N_boundary / N_total if N_total > 0 else 0

        # General 區域正確率
        if N_general > 0:
            general_correct = (labels[general_flags] == preds[general_flags]).sum()
            general_acc = general_correct / N_general
        else:
            general_acc = 0

        score = alpha * boundary_ratio + beta * general_acc
        print(f"{delta}: {boundary_ratio}, {general_acc}, {score}")
        if score > best_score:
            best_score = score
            best_delta = delta

    return best_delta

# add for cd distribution
def plot_cd_quadrant_distribution(df, class_names=['eMBB', 'mMTC', 'URLLC'], parent_dir=None):
    # 只保留 CD > 0 的樣本
    df_cd_pos = df[df['confidence_difference'] > 0].copy()

    # 加上每個樣本的類別名稱
    df_cd_pos['class_name'] = df_cd_pos['true_label'].apply(lambda x: class_names[x])

    # 設定 bin 與範圍
    bins = np.linspace(0, 1, 50)

    # 建立三類的 histogram，並正規化為比例
    hist_data = {}
    for cname in class_names:
        cds = df_cd_pos[df_cd_pos['class_name'] == cname]['confidence_difference']
        hist, _ = np.histogram(cds, bins=bins, density=False)
        hist_data[cname] = hist #/ len(df_cd_pos)  # normalize by total positive CD count

    # 將每個類別的 bar 放置在對應象限
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 8))
    width = 0.02  # bar 寬度

    # eMBB - 第一象限 (x > 0, y > 0)
    ax.bar(bin_centers, hist_data['eMBB'], width=width, color='tab:blue', label='eMBB')

    # mMTC - 第二象限 (x < 0, y > 0)
    ax.bar(-bin_centers, hist_data['mMTC'], width=width, color='tab:orange', label='mMTC')

    # URLLC - 第四象限 (x > 0, y < 0)
    ax.bar(bin_centers, -hist_data['URLLC'], width=width, color='tab:green', label='URLLC')

    # 加上輔助線與標籤
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Confidence Difference') # (0 → low, 1 → high)
    ax.set_ylabel('Sample Ratio') # (Positive: eMBB/mMTC, Negative: URLLC)
    ax.set_title('Confidence Difference Distribution by Class (CD > 0 only)')
    ax.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 儲存圖像
    quad_plot_path = os.path.join(parent_dir, 'cd_quadrant_distribution.png')
    plt.savefig(quad_plot_path)
    plt.close()
    print(f"CD quadrant distribution plot saved at {quad_plot_path}")

def eval_func(logdir, model_name, parent_dir=None, train_config=None, ds_info=None, ds_test=None, device=None, use_diffusion_data=False):
    # Load config and dataset info
    if train_config is None:
        # 讀取後還原
        with open(os.path.join(logdir, "train_config.json"), "r") as f: # modify for slice 1
        # with open(os.path.join("train_log1\slice__model_train_result", "train_config.json"), "r") as f:
            train_config = restore_train_config(json.load(f))
    if ds_info is None:
        with open(os.path.join(logdir, "ds_info.json"), "r") as f: # modify for slice 1
        # with open(os.path.join("train_log1\slice__model_train_result", "ds_info.json"), "r") as f:
            ds_info = json.load(f)
    if ds_test is None:
        # if use_diffusion_data:
        dataset_filename, norm_filename, dataset, cols_maxmin = convert_npy_to_pickle(parent_dir, use_diffusion_data) # modify for slice 1
        ds_test = TracesDataset(dataset_pkls=dataset_filename, key='train', sanitize=False, norm_par_path=norm_filename)
        # else:
        # ds_test = TracesDataset(dataset_pkls='colosseum_dataset_globalnorm.pkl', key='valid', path='data\colosseum\pickle_to_csv', sanitize=False, norm_par_path='colosseum_global_cols_maxmin.pkl')
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild the model and load the weight
    global_model = train_config['global_model']
    Nclass       = train_config['Nclass']
    num_feats    = train_config['num_feats']
    slice_len    = 1#train_config['slice_len']
    pos_enc      = train_config.get('pos_enc', False)

    # Load the model
    model, loss_fn = TRACTOR_model(Nclass, global_model, num_feats, slice_len, pos_enc)
    # Load the trained model weights
    model.load_state_dict(torch.load(os.path.join(logdir, model_name), map_location=device)['model_state_dict']) # modify for slice 1
    # model.load_state_dict(torch.load(os.path.join("train_log1\slice__model_train_result", "model.1.trans_v1.pt"), map_location=device)['model_state_dict'])
    # Move the model to the device and set it to eval mode
    model.to(device)
    model.eval()    

    ###################### TESTING WITH VALIDATION DATA #########################
    print(f"Test Analysis for model {train_config['model_postfix']} with slice {ds_info['slice_len']}:")
    # Num. params
    total_params = sum(p.numel() for p in model.parameters())
    print(f'TOTAL params        {total_params}')

    test_dataloader = DataLoader(ds_test, batch_size=train_config['batch_size'], shuffle=False)

    size = len(test_dataloader.dataset)
    correct = 0
    conf_matrix = np.zeros((train_config['Nclass'], train_config['Nclass']))
    all_labels = []
    all_preds = []
    confidence_differences = [] # add for CD
    all_class_probs = [] # add for CD
    import torch.nn.functional as F # add for CD
    all_inputs = [] # add for boundary inputs
    all_true_labels = [] # add for boundary inputs

    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device)
            pred = model(X)


            # add for CD
            # Softmax後取 top-2 值並計算 confidence difference
            # softmax_out = F.softmax(pred, dim=1)
            # correct_class_probs = softmax_out[range(len(y)), y]  # 對應 ground truth 的機率
            # sum_other_probs = softmax_out.sum(dim=1) - correct_class_probs
            # conf_diff_batch = (correct_class_probs - sum_other_probs).cpu().numpy()

            logits = pred
            true_logit = logits[torch.arange(logits.size(0)), y]
            logits_clone = logits.clone()
            logits_clone[torch.arange(logits.size(0)), y] = float('-inf')
            max_other_logit, _ = logits_clone.max(dim=1)

            confidence_diff_logit = true_logit - max_other_logit
            confidence_differences.extend(confidence_diff_logit.cpu().numpy())
            probs = torch.softmax(pred, dim=1)  # 機率分佈
            all_class_probs.extend(probs.cpu().numpy().tolist())


            correct += (pred.cpu().argmax(1) == y).type(torch.float).sum().item()
            conf_matrix += conf_mat(y, pred.cpu().argmax(1), labels=list(range(train_config['Nclass'])))

            all_labels.extend(y.numpy().tolist())   # add for f1-score # 把每個batch的y加到all_labels
            all_preds.extend(pred.cpu().argmax(dim=1).numpy().tolist()) # add for f1-score # 把每個batch的pred加到all_preds # 0429 modify
            all_inputs.extend(X.cpu().numpy()) # add for boundary inputs
            all_true_labels.extend(y.cpu().numpy())# add for boundary inputs
    correct /= size
    # Accuracy
    print(
        f"Test Error: \n "
        f"Accuracy: {(100 * correct):>0.2f}%"
    )
    # Conf. Matrix
    conf_matrix = conf_matrix.astype('float')
    for r in range(conf_matrix.shape[0]):  # for each row in the confusion matrix
        sum_row = np.sum(conf_matrix[r, :])
        conf_matrix[r, :] = conf_matrix[r, :] / sum_row  * 100. # compute in percentage
    axis_lbl = ['eMBB', 'mMTC', 'URLLC'] 
    df_cm = pd.DataFrame(conf_matrix, axis_lbl, axis_lbl)
    # plt.figure(figsize=(10,7))
    sn.set(font_scale=1.8)  # for label size
    sn.heatmap(df_cm, vmin=0, vmax=100, annot=True, cmap=sn.color_palette("light:b", as_cmap=True), annot_kws={"size": 25}, fmt='.1f')  # font size
    name_suffix = '_diffusion' if use_diffusion_data else ''
    add_ctrl = ''
    confusion_path = os.path.join(parent_dir, f"Results_slice_{ds_info['slice_len']}.{train_config['model_postfix']}{add_ctrl}{name_suffix}_test.png")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=200, bbox_inches='tight')
    print(f'Confusion matrix saved to {confusion_path}')
    # plt.show()
    plt.close()
    print('-------------------------------------------')
    print('Global confusion matrix (%) (validation split)')
    print(conf_matrix)
    # Inference time analysis
    inputs, _ = next(iter(test_dataloader))  
    sample_input = torch.unsqueeze(inputs[0], 0)
    sample_input = sample_input.to(device)
    m, sd = timing_inference_GPU(sample_input, model)
    print('-------------------------------------------')
    print('Inference time analysis:')
    print(f'Mean: {m}, Standard deviation: {sd}')

    # After evaluation
    # Return metrics for tuning or logging
    raw_metrics = {
        'accuracy': correct * 100,
        'confusion_matrix': conf_matrix.tolist(),
        'total_params': total_params,
        'inference_time_mean': m,
        'inference_time_std': sd,
    }

    # add for f1-score
    # Calculate f1_macro
    f1_macro_score = f1_score(all_labels, all_preds, average='macro')

    # Wrap for MetricsReport
    metrics_report_input = {
        "test": {
            "accuracy": raw_metrics["accuracy"] / 100,
            "macro avg": {
                "f1-score": f1_macro_score,
            },
        }
    }

    # Create report and save
    report = tddpm_lib.MetricsReport(metrics_report_input, tddpm_lib.TaskType.MULTICLASS)
    tddpm_lib.dump_json(report.to_dict(), os.path.join(parent_dir, "results_eval.json"))
    print(f'results_eval save to {parent_dir}')

    # (Optional) print out nicely
    # report.print_metrics()


    # add for CD
    # ==== 儲存 CD 結果為 CSV ====
    class_names = ['eMBB', 'mMTC', 'URLLC']
    # 擷取每類機率作為獨立欄位
    class_probs_columns = list(zip(*all_class_probs))  # 轉置後變成每一類一欄
    # delta = find_best_delta(confidence_differences, all_labels, all_preds, alpha=0.1, beta=0.9)
    delta = 0.35  # 自訂邊界範圍 δ
    boundary_zone_flags = [int(abs(cd) <= delta) for cd in confidence_differences]
    df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'confidence_difference': confidence_differences,
        'boundary_zone_flag': boundary_zone_flags,
        class_names[0] + '_prob': class_probs_columns[0],
        class_names[1] + '_prob': class_probs_columns[1],
        class_names[2] + '_prob': class_probs_columns[2],
    })
    csv_path = os.path.join(parent_dir, 'eval_confidence_details.csv')
    df.to_csv(csv_path, index=False)
    print(f"CD results saved to CSV at {csv_path}")

    cd_values = np.asarray(confidence_differences, dtype=float)
    # 畫 histogram
    plt.figure(figsize=(8, 5))
    plt.hist(cd_values, bins=100, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='CD = 0')
    plt.axvline(+1.0, color='green', linestyle='--', label='delta = +1.0')
    plt.axvline(-1.0, color='green', linestyle='--', label='delta = -1.0')
    plt.title("Histogram of Confidence Difference (logit-based)")
    plt.xlabel("Confidence Difference (true_logit - max_other_logit)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    histogram_path = os.path.join(
        parent_dir,
        'confidence_difference_histogram.png',
    )
    plt.savefig(histogram_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'CD histogram saved to {histogram_path}')
    # ==== 繪製並儲存 CD 圖 ====
    save_cd_distribution_plot(confidence_differences, delta, parent_dir)

    # ==== 統計 CD ∈ [−δ, +δ] 區間的正確與錯誤分類比例 ====
    # 篩選 CD 落在 ±delta 之間的樣本
    boundary_df = df[df['boundary_zone_flag'] == 1]

    # 計算總樣本數與分類正確數
    total_boundary = len(boundary_df)
    correct_boundary = (boundary_df['true_label'] == boundary_df['predicted_label']).sum()
    incorrect_boundary = total_boundary - correct_boundary

    # 計算比例
    correct_ratio = correct_boundary / total_boundary if total_boundary > 0 else 0
    incorrect_ratio = incorrect_boundary / total_boundary if total_boundary > 0 else 0

    # 印出結果
    print(f"Total number of samples in boundary zone [±{delta}]: {total_boundary}")
    print(f" - Number of CORRECT classifications: {correct_boundary} ({correct_ratio:.2%})")
    print(f" - Number of INCORRECT classifications: {incorrect_boundary} ({incorrect_ratio:.2%})")

    general_df = df[(df['boundary_zone_flag'] == 0) & (df['true_label'] == df['predicted_label'])]
    out_of_boundary_df = df[(df['boundary_zone_flag'] == 0) & (df['true_label'] != df['predicted_label'])]
    general_correct = (general_df['true_label'] == general_df['predicted_label']).sum()
    out_of_boundary_correct = (out_of_boundary_df['true_label'] == out_of_boundary_df['predicted_label']).sum()
    general_ratio = general_correct / len(general_df) if len(general_df) else 0.0
    out_of_boundary_ratio = (
        out_of_boundary_correct / len(out_of_boundary_df)
        if len(out_of_boundary_df)
        else 0.0
    )
    print(f"general zone has {len(general_df)} samples, and {general_ratio:.2%} are correct.")
    print(
        f"out of boundary zone has {len(out_of_boundary_df)} samples, "
        f"and {out_of_boundary_ratio:.2%} are correct."
    )
    total_samples = len(out_of_boundary_df) + len(boundary_df) + len(general_df)
    print(f"zone sample ratio(out-of-boundary|in-boundary|general): {len(out_of_boundary_df) / total_samples:.2%} | {len(boundary_df) / total_samples:.2%} | {len(general_df) / total_samples:.2%}")

    # add for boundary inputs
    # ====== Saving boundary inputs ======
    # 將 boundary zone 中的資料輸入從 all_inputs 中取出
    boundary_mask = np.array(boundary_zone_flags) == 1

    boundary_inputs = np.array(all_inputs)[boundary_mask]
    boundary_labels = np.array(all_true_labels)[boundary_mask]

    # 儲存為 .npy 檔案
    np.save(os.path.join(parent_dir, 'boundary_zone_inputs.npy'), boundary_inputs)
    np.save(os.path.join(parent_dir, 'boundary_zone_labels.npy'), boundary_labels)

    print("Saved boundary zone input and label .npy files:")
    print(f"   Inputs → {os.path.join(parent_dir, 'boundary_zone_inputs.npy')}")
    print(f"   Labels → {os.path.join(parent_dir, 'boundary_zone_labels.npy')}")

    # ====== Saving cd positive inputs ======
    non_boundary_positive_cd_mask = (np.array(boundary_zone_flags) == 0) & (np.array(confidence_differences) > 0)
    non_boundary_inputs = np.array(all_inputs)[non_boundary_positive_cd_mask]
    non_boundary_labels = np.array(all_true_labels)[non_boundary_positive_cd_mask]

    np.save(os.path.join(parent_dir, 'non_boundary_positive_cd_inputs.npy'), non_boundary_inputs)
    np.save(os.path.join(parent_dir, 'non_boundary_positive_cd_labels.npy'), non_boundary_labels)

    print("Saved non-boundary but positive CD input and label .npy files:")
    print(f"   Inputs → {os.path.join(parent_dir, 'non_boundary_positive_cd_inputs.npy')}")
    print(f"   Labels → {os.path.join(parent_dir, 'non_boundary_positive_cd_labels.npy')}")
    # ====== CD > 0 分佈視覺化（象限圖）======
    # add for cd distribution
    plot_cd_quadrant_distribution(df, class_names=['eMBB', 'mMTC', 'URLLC'], parent_dir=parent_dir)
    
    return report

def TRACTOR_model(Nclass, global_model, num_feats, slice_len, pos_enc=False, dropout=0.25):
    # Create model.
    if global_model in [TransformerNN, TransformerNN_v2]:
        model = global_model(classes=Nclass, slice_len=slice_len, num_feats=num_feats, use_pos=pos_enc, nhead=1,
                             custom_enc=True)
        loss_fn = nn.NLLLoss()
    elif global_model == ViT:
        model = global_model(classes=Nclass, slice_len=slice_len, num_feats=num_feats, dropout=dropout)
        # this loss fn is different (cause model doesn't have Softmax)
        loss_fn = nn.CrossEntropyLoss()
        # let's also add a transform function to add the channel axis for visual transformer in dataset samples
        ds_train.transform = add_first_dim
        ds_test.transform = add_first_dim
    else:
        model = global_model(classes=Nclass, slice_len=slice_len, num_feats=num_feats)
        loss_fn = nn.NLLLoss()
    return  model, loss_fn

def add_first_dim(x):
    return x[None]

def timing_inference_GPU(dummy_input, model):
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn

def prepare_train_config_for_saving(train_config):
    config = train_config.copy()  # 避免改到原本的 dict

    # 將 global_model 類別轉換成字串
    if "global_model" in config and not isinstance(config["global_model"], str):
        config["global_model"] = config["global_model"].__name__
    
    # 將 device 轉換成字串
    if "device" in config and not isinstance(config["device"], str):
        config["device"] = str(config["device"])

    return config

def restore_train_config(train_config):
    # 還原 global_model
    if "global_model" in train_config and isinstance(train_config["global_model"], str):
        train_config["global_model"] = getattr(classifier_models, train_config["global_model"])
    
    # 還原 device
    if "device" in train_config and isinstance(train_config["device"], str):
        train_config["device"] = torch.device(train_config["device"])
    
    return train_config

import json
from json_tricks import dump
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_file", nargs='+', required=True, help="Name of dataset pickle file containing training data and labels.")
    parser.add_argument("--ds_path", default="../logs/", help="Specify path where dataset files are stored")
    parser.add_argument("--isNorm", default=False, action='store_true', help="Specify to load the normalized dataset." )
    parser.add_argument("--norm_param_path", default="", help="Normalization parameters path.")
    parser.add_argument("--mode", default="train", choices=['train', 'test'], help="Testing the model") # TODO visualize capture and then perform classification after loading model
    parser.add_argument("--exp_name", default='', help="Name of this experiment")
    parser.add_argument("--transformer", default=None, choices=['v1', 'v2', 'ViT'], help="Use Transformer based model instead of CNN, choose v1 or v2 ([CLS] token)")
    parser.add_argument("--dropout", default=0.25, type=float, help="Only used for Visual Transformer (ViT)")
    parser.add_argument("--pos_enc",  action="store_true", default=False, help="Use positional encoder (only applied to transformer arch)")
    parser.add_argument("--patience", type=int, default=30, help="Num of epochs to wait before interrupting training with early stopping")
    parser.add_argument("--lrmax", type=float, default=1e-3,help="Initial learning rate ")
    parser.add_argument("--lrmin", type=float, default=1e-5, help="Final learning rate after scheduling ")
    parser.add_argument("--lrpatience", type=int, default=10, help="Patience before triggering learning rate decrease")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="[Deprecated] Enables GPU training")
    parser.add_argument("--log_base_dir", default="./train_log6", help="Specify base directory for saving logs and models.") # add
    parser.add_argument("--slicelen", default=1, type=int, help="Specify the slices lengths while generating the dataset.")
    parser.add_argument("--data_path", default="../data/colosseum", help="Specify path where numpy dataset files are stored")

    # Parse arguments (parser) and ignore unknown arguments
    args, _ = parser.parse_known_args()

    # ===== Select model =====
    # Select the version of the Transformer model
    transformer = {'v1': TransformerNN, 'v2': TransformerNN_v2, 'ViT': ViT}.get(args.transformer, None)
    
    slicelen = args.slicelen

    # ===== Load Datasets =====
    dataset_filename, norm_filename, dataset, cols_maxmin = convert_npy_to_pickle(data_dir=args.data_path, output_dir=args.ds_path, dataset_filename=args.ds_file, norm_filename=args.norm_param_path)

    # Load the training dataset
    print("--- Loading Train dataset...")
    ds_train = TracesDataset(dataset_pkls=args.ds_file, key='train', sanitize=False, norm_par_path=args.norm_param_path)
    # Load the validation dataset
    print("--- Loading Validation dataset...")
    ds_test  = TracesDataset(dataset_pkls=args.ds_file, key='valid', sanitize=False, norm_par_path=args.norm_param_path)
    
    print(np.bincount([lbl.item() for lbl in ds_train.obs_labels]))

    # Get basic information about the dataset
    ds_info = ds_train.info()

    print("--- DS INFO ---")
    print(ds_info)

    # ===== Set training parameters =====
    # Create a folder to store models and training records
    logdir = os.path.join(args.log_base_dir, args.exp_name) # modify # logdir = os.path.join(args.cp_path, args.exp_name)
    os.makedirs(logdir, exist_ok=True)
    # Set the computing device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA is available:", torch.cuda.is_available(), " Device:", device)

    # Load normalize parameters
    normp = pickle.load(open(os.path.join(args.ds_path, args.norm_param_path), "rb"))

    train_config = {
        # Set learning rate related parameters
        "lr": args.lrmax,
        "lrmin": args.lrmin,
        "lrpatience": args.lrpatience,
        "batch_size": 512,
        "epochs": 350,
        # Set training related parameters
        "Nclass": ds_info['nclasses'],
        "slice_len": ds_info['slice_len'],
        "num_feats": ds_info['numfeats'],
        "logdir": logdir,
        "pos_enc": args.pos_enc,
        "patience": args.patience,
        "dropout": args.dropout,
        # Set other parameters
        "global_model": ConvNN if args.transformer is None else transformer,
        "model_postfix": f"trans_{args.transformer}" if args.transformer else "cnn",
        "device": device
    }

    # If it is not test mode, start training
    if args.mode == 'train':
        # Single machine training
        loss_result, model_name = train_func(train_config)

    train_config_to_save = prepare_train_config_for_saving(train_config)
    # Save config and dataset info
    with open(os.path.join(logdir, "train_config.json"), "w") as f:
        json.dump(train_config_to_save, f)
    with open(os.path.join(logdir, "ds_info.json"), "w") as f:
        dump(ds_info, f, allow_nan=True)
    torch.save(ds_test, os.path.join(logdir, "test_dataset.pt"))

    # Test mode
    eval_func(logdir, model_name, train_config, ds_info, ds_test, device)


