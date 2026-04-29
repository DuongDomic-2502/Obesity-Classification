import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# ======================
# 1. Load data
# ======================
df_pca = pd.read_csv(r"D:\MachineLearning\BTL\data\Scaled_PCA_LDA\pca_data.csv")
df_lda = pd.read_csv(r"D:\MachineLearning\BTL\data\Scaled_PCA_LDA\lda_data.csv")

# ======================
# 2. Encode label (1 lần duy nhất)
# ======================
le = LabelEncoder()
le.fit(df_pca['NObeyesdad'])

datasets = {
    "PCA": (
        df_pca[['PC1', 'PC2']].values,
        le.transform(df_pca['NObeyesdad'].values)
    ),
    "LDA": (
        df_lda[['LD1', 'LD2']].values,
        le.transform(df_lda['NObeyesdad'].values)
    ),
}

colors = plt.cm.tab10(np.linspace(0, 1, len(le.classes_)))

# ======================
# 3. Plot
# ======================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, (name, (X, y)) in zip(axes, datasets.items()):

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = SVC(kernel='linear', C=1, gamma='scale')
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name} - Linear SVM accuracy: {acc:.4f}")

    # Mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Decision boundary
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='tab10')

    # Train points
    for i, color in zip(np.unique(y), colors):
        ax.scatter(
            X_train[y_train == i, 0],
            X_train[y_train == i, 1],
            color=color,
            edgecolor='k',
            s=40,
            label=f"Train - {le.inverse_transform([i])[0]}"
        )

    # Test points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap='tab10',
        marker='^',
        edgecolors='red',
        linewidths=1,
        s=60,
        label='Test'
    )

    col1, col2 = ('PC1', 'PC2') if name == 'PCA' else ('LD1', 'LD2')
    ax.set_title(f"SVM Decision Boundary (Linear) - {name}\nAccuracy: {acc:.2%}")
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.show()