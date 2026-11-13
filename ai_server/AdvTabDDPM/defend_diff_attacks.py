# defend_diff_attacks.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe
import os

def plot_defense_accuracy():
    # 攻擊種類
    attacks = ['FGSM', 'PGD', 'C&W']
    
    # 不同模型正確率
    acc_original = [59.07, 51.73, 44.76]   # 原始模型
    acc_at = [79.08, 79.43, 73.41]         # Adversarial Training
    acc_lu_at = [83.51, 84.15, 82.58]      # Label Uncertainty AT
    
    x = np.arange(len(attacks))
    width = 0.25
    
    # Seaborn風格
    sns.set(style="whitegrid")
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14
    })
    
    fig, ax = plt.subplots(figsize=(10,6))
    
    # 顏色設定
    color_original = '#d62728'  # 紅色
    color_at = '#1f77b4'        # 藍色
    color_lu_at = '#2ca02c'     # 綠色
    
    # 畫柱狀圖
    bars1 = ax.bar(x - width, acc_original, width, label='Total Accuracy (Before AT)', color=color_original, alpha=0.7)
    bars2 = ax.bar(x, acc_at, width, label='AT-Total Accuracy', color=color_at, alpha=0.7)
    bars3 = ax.bar(x + width, acc_lu_at, width, label='LU_AT-Total Accuracy', color=color_lu_at, alpha=0.7, hatch='//')
    
    # 標註柱狀圖數值
    for i in range(len(attacks)):
        ax.text(x[i] - width, acc_original[i] + 1, f'{acc_original[i]:.2f}%', ha='center', va='bottom', fontsize=12)
        ax.text(x[i], acc_at[i] + 3, f'{acc_at[i]:.2f}%', ha='center', va='bottom', fontsize=12)
        ax.text(x[i] + width, acc_lu_at[i] + 1, f'{acc_lu_at[i]:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 折線圖顯示提升差異
    diff_at = [acc_at[i] - acc_original[i] for i in range(len(acc_original))]
    diff_lu_at = [acc_lu_at[i] - acc_original[i] for i in range(len(acc_original))]
    
    ax2 = ax.twinx()  # 共享x軸，建立第二y軸
    line1, = ax2.plot(x, diff_at, color=color_at, marker='o', linestyle='-.', label='AT Accuracy Gain', linewidth=2)
    line2, = ax2.plot(x, diff_lu_at, color=color_lu_at, marker='s', linestyle='-.', label='LU_AT Accuracy Gain', linewidth=2)
    ax2.set_ylabel('Accuracy Gain (%)')
    # ax2.set_ylim(0, max(max(diff_at), max(diff_lu_at)) + 10)
    ax2.set_ylim(0, 50)
    
    # 標註折線圖數值（小字，不遮柱狀圖）
    for i in range(len(x)):
        ax2.text(
            x[i], diff_at[i] - 2.5, f'{diff_at[i]:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=color_at,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )
        ax2.text(
            x[i], diff_lu_at[i] - 2.5, f'{diff_lu_at[i]:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=color_lu_at,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )
    
    # 座標軸與標題
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Accuracy (%)')
    # ax.set_title('Model Accuracy Against Different Adversarial Attacks')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylim(0, 100)
    
    # 圖例
    bars_legend = ax.legend(loc='lower right')
    lines_legend = ax2.legend(handles=[line1, line2], loc='upper left', bbox_to_anchor=(0, 1.08))
    ax.add_artist(bars_legend)

    # 加入 Clean Data Baseline
    clean_acc = 91.98
    ax.axhline(clean_acc, color='purple', linestyle='--', linewidth=2, label='Clean Data Accuracy (After AT)')
    ax.text(len(attacks) - 0.5, clean_acc + 0.5, f'{clean_acc:.2f}%', color='purple', fontsize=15, fontweight='bold')
    # 調整圖例避免重疊
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    # 儲存圖片到0628資料夾
    os.makedirs('0628', exist_ok=True)
    plt.savefig('0628/defense_accuracy_comparison.png', dpi=300)
    
    plt.show()

if __name__ == "__main__":
    plot_defense_accuracy()
