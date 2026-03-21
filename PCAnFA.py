import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'D:\MachineLearning\BTL\ObesityDataSet_encoded.csv')

X = df.drop(columns=['NObeyesdad'])
y = df['NObeyesdad']


#BƯỚC 1: Chuẩn hóa dữ liệu (bắt buộc trước PCA và FA)
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# PCA
# ============================================================
pca = PCA()
pca.fit(X_scaled)

# Scree plot — chọn số components
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained)+1), explained, 'bo-')
plt.title('Scree Plot')
plt.xlabel('Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative)+1), cumulative, 'ro-')
plt.axhline(y=0.95, color='green', linestyle='--', label='95% variance')
plt.title('Cumulative Explained Variance')
plt.xlabel('Component')
plt.ylabel('Cumulative')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('pca_scree.png')
plt.show()

# Chọn số components giữ 95% variance
n_components_95 = np.argmax(cumulative >= 0.95) + 1
print(f'Số components giữ 95% variance: {n_components_95}')

# Áp dụng PCA
pca_final = PCA(n_components=n_components_95)
X_pca = pca_final.fit_transform(X_scaled)
print(f'Shape sau PCA: {X_pca.shape}')

# FA (Factor Analysis)

n_factors = n_components_95  # dùng cùng số với PCA để so sánh

fa = FactorAnalysis(n_components=n_factors, random_state=42)
X_fa = fa.fit_transform(X_scaled)
print(f'Shape sau FA: {X_fa.shape}')

# Xem factor loadings (mức độ ảnh hưởng của từng biến gốc)
loadings = pd.DataFrame(
    fa.components_.T,
    index=X.columns,
    columns=[f'Factor_{i+1}' for i in range(n_factors)]
)
print('\nFactor Loadings:')
print(loadings.round(3))

# So sánh PCA vs FA (visualize 2D)

pca_2d = PCA(n_components=2).fit_transform(X_scaled)
fa_2d  = FactorAnalysis(n_components=2, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y, cmap='tab10', alpha=0.5, s=10)
plt.title('PCA - 2D')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(scatter)

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(fa_2d[:, 0], fa_2d[:, 1], c=y, cmap='tab10', alpha=0.5, s=10)
plt.title('FA - 2D')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.colorbar(scatter2)

plt.tight_layout()
plt.savefig('pca_vs_fa.png')
plt.show()