import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from matplotlib.gridspec import GridSpec

# --- Đọc dữ liệu (đã scale sẵn) ---
df = pd.read_csv(r"D:\MachineLearning\BTL\data\scaled\data_scaled.csv")

# --- Tách X và y ---
X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']

# --- Encode label để vẽ màu ---
y_cat = pd.Categorical(y)
y_codes = y_cat.codes
class_names = y_cat.categories

# --- Data đã scale sẵn ---
X = X.values

# --- LDA ---
lda = LinearDiscriminantAnalysis(n_components=min(6, len(class_names) - 1))
X_lda = lda.fit_transform(X, y)

# --- Colormap ---
n_classes = len(class_names)
cmap = plt.cm.get_cmap('tab10', n_classes)
colors = [cmap(i) for i in range(n_classes)]

# --- Các cặp LD ---
pairs = [(0,1), (0,2), (0,3), (1,2), (1,3)]

# --- Tạo figure với GridSpec 3 hàng x 4 cột ---
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 4, figure=fig)

axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[1, 0:2]),
    fig.add_subplot(gs[1, 2:4]),
    fig.add_subplot(gs[2, 1:3]),  # subplot cuối căn giữa
]

for i, (a, b) in enumerate(pairs):
    ax = axes[i]
    ax.scatter(
        X_lda[:, a], X_lda[:, b],
        c=y_codes,
        cmap=cmap,
        vmin=0, vmax=n_classes - 1,
        alpha=0.7
    )
    ax.set_xlabel(f'LD{a+1}', fontsize=11)
    ax.set_ylabel(f'LD{b+1}', fontsize=11)
    ax.set_title(f'LD{a+1} vs LD{b+1}', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

# --- Legend chung ---
handles = [
    plt.Line2D([], [], marker='o', linestyle='', color=colors[i],
               markersize=8, label=int(class_names[i]))
    for i in range(n_classes)
]

fig.legend(handles=handles, title="Classes", bbox_to_anchor=(1.02, 0.95), loc='upper left')

plt.suptitle("LDA Visualization", fontsize=16)
plt.tight_layout()

# --- Lưu ảnh ---
plt.savefig(r"D:\MachineLearning\BTL\Plot\LDA_visualization.png", dpi=300, bbox_inches='tight')
plt.show()