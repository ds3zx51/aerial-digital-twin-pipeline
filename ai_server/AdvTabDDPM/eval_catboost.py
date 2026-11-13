from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import classification_report, r2_score
import numpy as np
import os
from sklearn.utils import shuffle
import zero
from pathlib import Path
import tddpm_lib
from pprint import pprint
from tddpm_lib import concat_features, read_pure_data, get_catboost_config, read_changed_val
# from sklearn.preprocessing import LabelEncoder # add for y_val label encode

# def ensure_consistent_dtypes(X):
#     """Ensure that all parts of the dataset have consistent data types."""
#     for part in ['train', 'val', 'test']:
#         print(f"{part} Data Types {X[part].dtype}")
#         # print(f"{part} NaNs {np.isnan(X[part]).sum()}")
#         if X['train'].dtype != X[part].dtype:
#             print(f"{X[part].dtype} Inconsistent data type for {part}. Converting to match 'train' {X['train'].dtype}.")
#             # Convert part to the same type as 'train'
#             X[part] = X[part].astype(X['train'].dtype)
#     return X
# add
def ensure_consistent_dtypes(X_num, X_cat, y):
    # Convert X_num to the desired float types
    if isinstance(X_num, np.ndarray):
        # Convert all elements to float32 if they are not already
        if not np.issubdtype(X_num.dtype, np.floating):
            X_num = X_num.astype(np.float64)
            # X_num = X_num.astype(np.float32)  # Convert to float32 if necessary
    else:
        raise TypeError("Expected X_num to be a numpy.ndarray")

    # Convert X_cat to the desired object and <U21 types
    if isinstance(X_cat, np.ndarray):
        if X_cat.dtype != object:  # Use 'object' instead of 'np.object'
            X_cat = X_cat.astype(object)  # Convert to object type

        # Ensure that the string types are <U21
        if X_cat.dtype == object:
            X_cat = X_cat.astype(np.int64)
            # X_cat = np.array([str(val) for val in X_cat], dtype='<U21')  # Convert to <U21
    else:
        raise TypeError("Expected X_cat to be a numpy.ndarray")
    # y is assumed to be consistent, but you can add checks if needed
    return X_num, X_cat, y

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
def visualize_and_export_synthetic_predictions(D, X, predictions, parent_dir, mode="cd_projection"):

    os.makedirs(parent_dir, exist_ok=True)

    X_val = X['train']
    y_true = D.y['train']
    y_prob = predictions['train']
    y_pred = np.argmax(y_prob, axis=1)

    label_names = ['eMBB', 'mMTC', 'URLLC']
    color_map = {0: 'red', 1: 'green', 2: 'blue'}

    if mode == "pca":
        print("[Info] Plotting PCA classification regions with predicted labels...")

        # PCA 降維
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_val)

        # 分類區域圖背景
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        h = max((x_max - x_min) / 300.0, (y_max - y_min) / 300.0)

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        pca_input = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
        Z = np.argmax(predictions['val'][np.random.choice(len(X_val), size=1)], axis=1)[0] * np.ones(xx.ravel().shape)  # dummy region
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.2, colors=['#ffcccc', '#ccffcc', '#ccccff'])

        for i in np.unique(y_pred):
            plt.scatter(
                X_pca[y_pred == i, 0],
                X_pca[y_pred == i, 1],
                label=label_names[i],
                alpha=0.6,
                s=10,
                c=color_map[i]
            )

        plt.title("Synthetic Data Classification and PCA Projection")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(parent_dir, "synthetic_pca_classification.png"))
        plt.close()

    elif mode == "cd_projection":
        confidence_differences = y_prob[np.arange(len(y_pred)), y_pred] - 1.45 #2 * y_prob[np.arange(len(y_pred)), y_pred] - 1
        abs_cd = np.abs(confidence_differences)
        valid_idx = valid_idx = (y_pred == y_true)  # 僅繪製預測正確者

        plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 1.0, 50)

        for i, name in enumerate(label_names):
            class_idx = (y_pred == i) & valid_idx
            if np.sum(class_idx) == 0:
                continue
            class_cd = abs_cd[class_idx]
            hist, _ = np.histogram(class_cd, bins=bins)
            hist = hist / np.sum(hist)  # Normalize to probability
            plt.plot(bins[:-1], hist, label=name, color=color_map[i], linewidth=2)

        plt.xlabel("Confidence Difference |CD| (Higher = More Confident)")
        plt.ylabel("Proportion of Samples")
        plt.title("Distribution of Confidence Difference by Predicted Class")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(parent_dir, "confidence_distribution.png"))
        plt.close()

    elif mode == "constellation":
        from scipy.stats import entropy

        # 計算 entropy (normalized by log_3)
        log_base = y_prob.shape[1]
        sample_entropy = entropy(y_prob.T, base=log_base)

        # 只保留預測正確的樣本
        valid_idx = (y_pred == y_true)

        plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 1, 50)
        color_map = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, name in enumerate(label_names):
            class_idx = (y_pred == i) & valid_idx
            if np.sum(class_idx) == 0:
                continue
            ent_vals = sample_entropy[class_idx]
            hist, _ = np.histogram(ent_vals, bins=bins)
            hist = hist / np.sum(hist)  # normalize
            plt.plot(bins[:-1], hist, label=name, color=color_map[i], linewidth=2)

        plt.xlabel("Entropy (normalized, base 3)")
        plt.ylabel("Proportion")
        plt.title("Entropy Distribution (Correct Predictions Only)")
        plt.legend()
        plt.grid(True)

        os.makedirs(parent_dir, exist_ok=True)
        plt.savefig(os.path.join(parent_dir, "entropy_distribution.png"))
        plt.close()

    # 匯出 CSV
    df_export = pd.DataFrame()
    df_export['true_label'] = y_true
    df_export['pred_label'] = y_pred
    df_export['correct'] = (y_true == y_pred).astype(int)
    print("predict not correct samples: ", np.sum((y_true != y_pred).astype(int)))
    for i, name in enumerate(label_names):
        df_export[f'prob_{name}'] = y_prob[:, i]
    if mode == "cd_projection":
        df_export['confidence_difference'] = confidence_differences
    elif mode == "constellation":
        df_export['entropy'] = sample_entropy
    elif mode == "pca":
        df_export['pca_x'] = X_pca[:, 0]
        df_export['pca_y'] = X_pca[:, 1]

    df_export.to_csv(os.path.join(parent_dir, f"synthetic_predictions_{mode}.csv"), index=False)
    print(f"[Info] Exported synthetic prediction results to CSV.")

def train_catboost(
    parent_dir,
    real_data_path,
    eval_type,
    T_dict,
    seed = 0,
    params = None,
    change_val = True,
    device = None # dummy
):
    zero.improve_reproducibility(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)
    info = tddpm_lib.load_json(os.path.join(real_data_path, 'info.json'))
    T = tddpm_lib.Transformations(**T_dict)
    
    if change_val:
        print('change val is True') # add
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(real_data_path, val_size=0.2)

    X = None
    print('-'*100)
    if eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(real_data_path)
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path)

        ###
        # dists = privacy_metrics(real_data_path, synthetic_data_path)
        # bad_fakes = dists.argsort()[:int(0.25 * len(y_fake))]
        # X_num_fake = np.delete(X_num_fake, bad_fakes, axis=0)
        # X_cat_fake = np.delete(X_cat_fake, bad_fakes, axis=0) if X_cat_fake is not None else None
        # y_fake = np.delete(y_fake, bad_fakes, axis=0)
        ###

        y = np.concatenate([y_real, y_fake], axis=0)

        X_num = None
        if X_num_real is not None:
            X_num = np.concatenate([X_num_real, X_num_fake], axis=0)

        X_cat = None
        if X_cat_real is not None:
            X_cat = np.concatenate([X_cat_real, X_cat_fake], axis=0)

    elif eval_type == 'synthetic':
        print(f'loading synthetic data: {synthetic_data_path}')
        X_num, X_cat, y = read_pure_data(synthetic_data_path)

    elif eval_type == 'real':
        print(f'loading real data: {real_data_path}')
        if not change_val:
            X_num, X_cat, y = read_pure_data(real_data_path)
    else:
        raise "Choose eval method"

    # Calculate n_classes
    info['n_classes'] = len(np.unique(y)) # add

    if not change_val:
        print('change_val: ', change_val) # add
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, 'val')
        # y_val_encoded = encoder.fit_transform(y_val) # add for y_val label encode
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, 'test')

    D = tddpm_lib.Dataset(
        {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
        {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
        {'train': y, 'val': y_val, 'test': y_test},
        {},
        tddpm_lib.TaskType(info['task_type']),
        info.get('n_classes')
    )

    D = tddpm_lib.transform_dataset(D, T, None)
    X = concat_features(D)
    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')

    if params is None:
        print('params is None') # add
        catboost_config = get_catboost_config(real_data_path, is_cv=True)
    else:
        print(f'params: {params}') # add
        catboost_config = params

    if 'cat_features' not in catboost_config:
        catboost_config['cat_features'] = list(range(D.n_num_features, D.n_features))

    for col in range(D.n_features):
        for split in X.keys():
            if col in catboost_config['cat_features']:
                X[split][col] = X[split][col].astype(str)
            else:
                X[split][col] = X[split][col].astype(float)
    
    if D.is_regression:
        model = CatBoostRegressor(
            **catboost_config,
            eval_metric='RMSE',
            random_seed=seed
        )
        predict = model.predict
    else:
        model = CatBoostClassifier(
            loss_function="MultiClass" if D.is_multiclass else "Logloss",
            **catboost_config,
            eval_metric='TotalF1',
            random_seed=seed,
            class_names=[str(i) for i in range(D.n_classes)] if D.is_multiclass else ["0", "1"]
        )
        predict = (
            model.predict_proba
            if D.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]
        )

    model.fit(
        X['train'], D.y['train'],
        eval_set=(X['val'], D.y['val']),
        verbose=100
    )
    predictions = {k: predict(v) for k, v in X.items()}
    print(predictions['train'].shape)

    # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    # import matplotlib.pyplot as plt
    # if not D.is_regression:
    #     # 取預測類別
    #     if D.is_multiclass:
    #         y_pred_val = np.argmax(predictions['val'], axis=1)
    #     else:
    #         y_pred_val = (predictions['val'] > 0.5).astype(int)

    #     y_true_val = D.y['val']

    #     # 計算與顯示混淆矩陣
    #     cm = confusion_matrix(y_true_val, y_pred_val)
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #     disp.plot(cmap=plt.cm.Blues)
    #     plt.title("Confusion Matrix (Validation Set)")
    #     # plt.show()
    #     plt.savefig(os.path.join(parent_dir, "confusion_matrix.png"))

    # from torch.nn.functional import softmax
    # import torch

    # # 注意：模型預測的是 logit，需先轉 softmax 機率
    # visualize_and_export_synthetic_predictions(D, X, predictions, parent_dir)


    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = real_data_path
    report['metrics'] = D.calculate_metrics(predictions,  None if D.is_regression else 'probs')

    metrics_report = tddpm_lib.MetricsReport(report['metrics'], D.task_type)
    metrics_report.print_metrics()

    if parent_dir is not None:
        tddpm_lib.dump_json(report, os.path.join(parent_dir, "results_eval.json"))

    return metrics_report
