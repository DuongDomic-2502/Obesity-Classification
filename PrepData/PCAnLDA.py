import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('D:\\MachineLearning\\BTL\\data\\scaled\\train_scaled.csv')

X = data.drop(columns=['NObeyesdad']).values
y = data['NObeyesdad'].astype(int).values

n_classes = len(np.unique(y))
labels = [f"Class {i}" for i in range(n_classes)]
colors = plt.cm.tab10(np.linspace(0, 0.7, n_classes))

# ── PCA visualization (2 components) ──────────────────────
pca = PCA(n_components=14)
X_pca = pca.fit_transform(X)

print("=== PCA ===")
print(f"Explained variance ratio : {pca.explained_variance_ratio_}")
print(f"Tổng phương sai giữ lại  : {pca.explained_variance_ratio_.sum():.3f}")

# ── LDA visualization ──────────────────────────────────────
lda = LinearDiscriminantAnalysis(n_components=5)
X_lda = lda.fit_transform(X, y)

print("\n=== LDA ===")
print(f"Explained variance ratio : {lda.explained_variance_ratio_}")
print(f"Tổng phương sai giữ lại  : {lda.explained_variance_ratio_.sum():.3f}")

# ── Scatter PCA vs LDA ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, X_red, title, xlabel, ylabel in zip(
    axes,
    [X_pca, X_lda],
    ["PCA", "LDA"],
    [f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", "LD1"],
    [f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", "LD2"],
):
    for i, (label, color) in enumerate(zip(labels, colors)):
        mask = y == i
        ax.scatter(X_red[mask, 0], X_red[mask, 1],
                   color=color, label=label, alpha=0.7, s=30, edgecolors='none')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=8, markerscale=1.5)
    ax.grid(True, alpha=0.3)
plt.suptitle("PCA vs LDA — NObeyesdad Dataset", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('pca_lda_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Elbow chart ────────────────────────────────────────────
pca_full = PCA().fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cumvar) + 1), cumvar, marker='o', color='steelblue')
plt.axhline(0.95, linestyle='--', color='red',   label='95% variance')
plt.axhline(0.90, linestyle='--', color='orange', label='90% variance')
plt.xlabel("Số components"); plt.ylabel("Cumulative explained variance")
plt.title("PCA — Elbow Chart"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_elbow.png', dpi=150, bbox_inches='tight')
plt.show()

n_95 = np.argmax(cumvar >= 0.95) + 1
n_90 = np.argmax(cumvar >= 0.90) + 1
print(f"\nPCA: cần {n_90} components để đạt 90%, {n_95} để đạt 95%")












#Sau khi giảm chiều, có 4 nhóm đánh giá chính:

#Nhóm 1 — Đánh giá chất lượng giảm chiều (không cần model)

#Explained Variance Ratio — PCA giữ lại bao nhiêu % thông tin. Thường yêu cầu ≥ 85%
#Scree Plot — đồ thị elbow để chọn số chiều tối ưu
#LDA Separation — trực quan hóa các class có tách biệt rõ sau LDA không (scatter plot LD1 vs LD2)


#Nhóm 2 — Đánh giá hiệu năng phân loại (quan trọng nhất)

#Accuracy, Precision, Recall, F1-score — dùng classification_report của sklearn
#Confusion Matrix — xem mô hình nhầm class nào
#Cross-validation — đánh giá tổng quát hóa (đã làm ở trên)


#Nhóm 3 — So sánh trước và sau giảm chiều

#So sánh accuracy baseline vs PCA vs LDA trực tiếp — đây là test thực tế nhất cho báo cáo


#Nhóm 4 — Kiểm định thống kê (nâng cao)

#t-test / Wilcoxon — kiểm tra xem sự chênh lệch accuracy giữa PCA và LDA có ý nghĩa thống kê không