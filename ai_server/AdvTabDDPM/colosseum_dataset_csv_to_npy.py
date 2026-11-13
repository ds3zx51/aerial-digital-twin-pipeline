import numpy as np
from sklearn.model_selection import train_test_split, KFold

import os
import pandas as pd
import glob

import os
import pandas as pd
import glob

# 原始資料夾
input_folder = 'data/colosseum'

# 搜尋所有符合條件的CSV檔案
pattern = os.path.join(input_folder, 'sched*_tr*_exp*_bs*_slices_bs*_*_*_metrics.csv')
csv_files = glob.glob(pattern)

# 定義 y_feature
y_feature = 'traffic_type'

# 定義數值與類別特徵
numerical_features = [
    'Timestamp', 'num_ues', 'slice_prb', 'power_multiplier', 'dl_mcs', 'dl_n_samples',
    'dl_buffer [bytes]', 'tx_brate downlink [Mbps]', 'tx_pkts downlink',
    'tx_errors downlink (%)', 'dl_cqi', 'ul_mcs', 'ul_n_samples',
    'ul_buffer [bytes]', 'rx_brate uplink [Mbps]', 'rx_pkts uplink',
    'rx_errors uplink (%)', 'ul_rssi', 'ul_sinr', 'phr',
    'sum_requested_prbs', 'sum_granted_prbs', 'dl_pmi', 'dl_ri',
    'ul_n', 'ul_turbo_iters'
]

categorical_features = [
    'IMSI', 'RNTI', 'slicing_enabled', 'slice_id', 'scheduling_policy'
]

# 選出的特徵
selected_columns = [y_feature] + numerical_features + categorical_features

# 儲存結果用的資料夾與檔案
output_folder = os.path.join(input_folder, 'selected_feature')
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'colosseum_dataset_selected_features_metrics.csv')

# 儲存所有資料的 list
all_dfs = []

# 逐一讀取檔案並選取欄位
for file_path in sorted(csv_files):
    df = pd.read_csv(file_path)

    # 確保所有需要的欄位都在
    if all(col in df.columns for col in selected_columns):
        df_selected = df[selected_columns]
        all_dfs.append(df_selected)
    else:
        missing = [col for col in selected_columns if col not in df.columns]
        print(f"檔案 {os.path.basename(file_path)} 缺少欄位 {missing}，略過")

# 合併並儲存
if all_dfs:
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"所有資料已整合並儲存至：{output_path}")
else:
    print("沒有符合條件的資料被儲存。")


# Helper function to load data from CSV and separate features and labels
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X_num = data[numerical_features].values
    X_cat = data[categorical_features].astype(str).values
    # y = data['traffic'].values
    y = data[y_feature].values
    return X_num, X_cat, y

# Function for proportional split
def proportional_split(X_num, X_cat, y, train_size=0.7, val_size=0.15, test_size=0.15):
    # Split into train and temp (val + test)
    X_num_train, X_num_temp, X_cat_train, X_cat_temp, y_train, y_temp = train_test_split(
        X_num, X_cat, y, test_size=(val_size + test_size), random_state=42)
    # Split temp into val and test
    test_ratio = test_size / (val_size + test_size)
    X_num_val, X_num_test, X_cat_val, X_cat_test, y_val, y_test = train_test_split(
        X_num_temp, X_cat_temp, y_temp, test_size=test_ratio, random_state=42)
    return (X_num_train, X_cat_train, y_train), (X_num_val, X_cat_val, y_val), (X_num_test, X_cat_test, y_test)

# Function for K-Fold split
def k_fold_split(X_num, X_cat, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = []
    for train_index, test_index in kf.split(X_num):
        X_num_train, X_num_test = X_num[train_index], X_num[test_index]
        X_cat_train, X_cat_test = X_cat[train_index], X_cat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        folds.append(((X_num_train, X_cat_train, y_train), (X_num_test, X_cat_test, y_test)))
    return folds

# Load data
# X_num, X_cat, y = load_data(file) # modify for choose features
X_num, X_cat, y = load_data(output_path)

# 統計 X_num 的範圍
num_feature_ranges = {numerical_features[i]: (X_num[:, i].min(), X_num[:, i].max()) for i in range(X_num.shape[1])}

# 統計 X_cat 每個類別特徵的種類數和具體類別值
cat_feature_categories = {categorical_features[i]: np.unique(X_cat[:, i]) for i in range(X_cat.shape[1])}
cat_feature_counts = {feature: len(values) for feature, values in cat_feature_categories.items()}

# 統計 y 的種類數和具體類別值
y_categories = np.unique(y)
y_count = len(y_categories)

# 輸出統計結果
print("=== 數值特徵 X_num 的範圍 ===")
for feature, (min_val, max_val) in num_feature_ranges.items():
    print(f"{feature}: 最小值 = {min_val}, 最大值 = {max_val}")
print()

print("=== 類別特徵 X_cat 的統計 ===")
for feature, values in cat_feature_categories.items():
    print(f"{feature}: 總共有 {cat_feature_counts[feature]} 種類，分別是 {values}")
print()

print("=== 標籤 y 的統計 ===")
print(f"{y_feature} 總共有 {y_count} 種類，分別是 {y_categories}")

# Choose split method: Proportional Split or K-Fold Split
split_method = "proportional"  # Set to either "proportional" or "k_fold"
if split_method == "proportional":
    # Proportional split
    (X_num_train, X_cat_train, y_train), (X_num_val, X_cat_val, y_val), (X_num_test, X_cat_test, y_test) = proportional_split(X_num, X_cat, y)
    
    splits = {
        'train': (X_num_train, X_cat_train, y_train),
        'val': (X_num_val, X_cat_val, y_val),
        'test': (X_num_test, X_cat_test, y_test)
    }
    
    # Save proportional split data to .npy
    for split_name, (X_num_split, X_cat_split, y_split) in splits.items():
        np.save(os.path.join(input_folder, f'X_num_{split_name}.npy'), X_num_split)
        np.save(os.path.join(input_folder, f'X_cat_{split_name}.npy'), X_cat_split)
        np.save(os.path.join(input_folder, f'y_{split_name}.npy'), y_split)

        # Calculate and print the information for each split
        d_in = len(numerical_features)
        # num_categories = [len(np.unique(X_num_split[:, i])) for i in range(X_num_split.shape[1])]
        num_feature_ranges = {numerical_features[i]: (X_num_split[:, i].min(), X_num_split[:, i].max()) for i in range(X_num_split.shape[1])}
        # unique_cats = [np.unique(X_cat_split[:, i]) for i in range(X_cat_split.shape[1])]
        cat_feature_categories = {categorical_features[i]: np.unique(X_cat_split[:, i]) for i in range(X_cat_split.shape[1])}
        cat_feature_counts = {feature: len(values) for feature, values in cat_feature_categories.items()}
        y_categories = np.unique(y_split)
        y_count = len(y_categories)

        print(f'=== {split_name.upper()} SET ===')
        print(f'd_in: {d_in}')
        # print(f'num_categories: {num_categories}')
        print(f'num_categories: {num_feature_ranges}')
        # print(f'Unique values in categorical features ({split_name}): {unique_cats}')
        print(f'Unique values in categorical features ({split_name}): {cat_feature_counts}')
        print(f'Unique values in y feature ({split_name}): {y_count}')
        print()
    
    print("Proportional split data has been saved to .npy files.")

elif split_method == "k_fold":
    # K-Fold split
    folds = k_fold_split(X_num, X_cat, y)
    
    # Save each fold to .npy files and print information
    for i, ((X_num_train, X_cat_train, y_train), (X_num_test, X_cat_test, y_test)) in enumerate(folds):
        # Save train and test sets for each fold
        np.save(os.path.join(input_folder, f'X_num_train_fold_{i+1}.npy'), X_num_train)
        np.save(os.path.join(input_folder, f'X_cat_train_fold_{i+1}.npy'), X_cat_train)
        np.save(os.path.join(input_folder, f'y_train_fold_{i+1}.npy'), y_train)
        np.save(os.path.join(input_folder, f'X_num_test_fold_{i+1}.npy'), X_num_test)
        np.save(os.path.join(input_folder, f'X_cat_test_fold_{i+1}.npy'), X_cat_test)
        np.save(os.path.join(input_folder, f'y_test_fold_{i+1}.npy'), y_test)

        # Calculate and print the information for the train and test sets in each fold
        for split_name, (X_num_split, X_cat_split, y_split) in zip(['train', 'test'], [(X_num_train, X_cat_train, y_train), (X_num_test, X_cat_test, y_test)]):
            d_in = len(numerical_features)
            num_categories = [len(np.unique(X_num_split[:, i])) for i in range(X_num_split.shape[1])]
            unique_cats = [np.unique(X_cat_split[:, i]) for i in range(X_cat_split.shape[1])]

            print(f'=== Fold {i+1} {split_name.upper()} SET ===')
            print(f'd_in: {d_in}')
            print(f'num_categories: {num_categories}')
            print(f'Unique values in categorical features ({split_name}): {unique_cats}')
            print()

    print(f"{len(folds)}-Fold split data has been saved to .npy files.")

