import seaborn as sns   
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train = ...
y_train = ...
X_val = ...
y_val = ...
feature_names = ...
#
# #plot correlation
#
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X_train).correlation

# Ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=feature_names, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

# ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
sns.heatmap(corr[dendro["leaves"], :][:, dendro["leaves"]], annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)

ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
plt.show()
cluster_ids = hierarchy.fcluster(dist_linkage, 0.2, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
print(cluster_ids)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
# selected_features = [0,2,4,5,6,7,8]
print(selected_features)
X_train_sel = X_train[:, selected_features]
X_val_sel = X_val[:, selected_features]

clf_sel = RandomForestClassifier(n_estimators=500, max_depth=8,random_state=0)
clf_sel.fit(X_train_sel, y_train)
print(
    "Accuracy on val data with features removed: {:.2f}".format(
        clf_sel.score(X_val_sel, y_val)
    )
)
y_pred_sel = clf_sel.predict(X_val_sel)
print('val features removed',classification_report(y_val, y_pred_sel))