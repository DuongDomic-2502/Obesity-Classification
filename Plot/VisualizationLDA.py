import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("D:\MachineLearning\BTL\data\lda_data.csv")
y_codes, class_names = pd.factorize(df['NObeyesdad'])
X = df[['LD1', 'LD2']].values
cmap = plt.cm.tab10

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], c=y_codes, cmap=cmap, alpha=0.7, s=30)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_title('LDA Visualization')
ax.legend(handles=[mpatches.Patch(color=cmap(i), label=f'Class {class_names[i]}') for i in range(len(class_names))])

plt.tight_layout()
plt.savefig('LDA_visualization.png', dpi=200, bbox_inches='tight')
plt.show()