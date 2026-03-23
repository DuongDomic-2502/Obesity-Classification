
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler   


df = pd.read_csv('D:\\MachineLearning\\BTL\\ObesityDataSet_encoded_v4.csv')

########################### kiểm tra các cột cần để lấy standerlize tốt nhất  ##########################################
cols = ['Age', 'Height', 'Weight', 'NCP', 'CH2O', 'FAF']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
# ============================================================
# 1. HISTOGRAM
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Phân phối các đặc trưng liên tục', fontsize=16, fontweight='bold', y=1.01)

for i, (col, color) in enumerate(zip(cols, colors)):
    ax = axes[i // 3][i % 3]
    skew_val = df[col].skew()

    ax.hist(df[col], bins=30, color=color, edgecolor='white', linewidth=0.5, alpha=0.85)

    # Đường mean và median
    ax.axvline(df[col].mean(),   color='red',    linestyle='--', linewidth=1.5, label=f'Mean: {df[col].mean():.2f}')
    ax.axvline(df[col].median(), color='black',  linestyle='-',  linewidth=1.5, label=f'Median: {df[col].median():.2f}')

    ax.set_title(f'{col}  (skew = {skew_val:.2f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Giá trị', fontsize=10)
    ax.set_ylabel('Số lượng', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('histogram.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 2. BOXPLOT
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Phân phối và Outlier các đặc trưng liên tục', fontsize=16, fontweight='bold', y=1.01)

for i, (col, color) in enumerate(zip(cols, colors)):
    ax = axes[i // 3][i % 3]

    bp = ax.boxplot(df[col], patch_artist=True, vert=True,
                    flierprops=dict(marker='o', markerfacecolor=color, markersize=4, alpha=0.5),
                    medianprops=dict(color='black', linewidth=2),
                    boxprops=dict(facecolor=color, alpha=0.6),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    # Thống kê
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)][col]

    ax.set_title(f'{col}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Giá trị', fontsize=10)
    ax.set_xlabel(f'Outliers: {len(outliers)} điểm  |  IQR: {iqr:.2f}', fontsize=9, color='gray')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig('boxplot.png', dpi=150, bbox_inches='tight')
plt.show()