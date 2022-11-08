#%%
from cProfile import label
import os
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns   
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

from sklearn.preprocessing import StandardScaler

def create_dataset(file_path='data/CBD_FEATURES'):
    label_dct = {'complex_swr':0,'swr':1,'ripple':2}
    df = pd.DataFrame(columns=['id','rat_id','label','MeanFreq','Amplitude','AUC','Duration','Peak2Peak','Power','Entropy','NumberOfPeaks','SpecEntropy'])
    #walk through files in folder
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.mat'):
                filename = os.path.join(root, file)
                print(filename)
                ratID = int(filename.split('_')[-2].split('ratID')[-1])
                mat = loadmat(filename)
                for k in ['complex_swr','swr','ripple']:
                    data = mat['GC_' + k + '_ratID'+ str(ratID) +'_cbd']['PCA_features'][0,0]
                    df_tmp = pd.DataFrame(data,columns=['MeanFreq','Amplitude','AUC','Duration','Peak2Peak','Power','Entropy','NumberOfPeaks','SpecEntropy'])
                    df_tmp['id'] = df_tmp.index
                    df_tmp['rat_id'] = ratID
                    df_tmp['label'] = label_dct[k]
                    df = pd.concat([df,df_tmp])
    df.to_csv('proc_data/veh_features.csv')
# %%
df = pd.read_csv('proc_data/veh_features.csv')
df = df[~df.rat_id.isin([10,204])]
acc_values = []
f1_values = []
#set seed
np.random.seed(42)
df.columns
#%%
#veh
folds = {1: [206, 210], 2:[203,211],3:[9,201],4:[213,4,210]}
#cbd
# folds = {1: [214, 205], 2:[5,2],3:[207,212],4:[11,209]}
feat_importance = pd.DataFrame()
log_prob = None
sel_features_calc = ['MeanFreq', 'AUC', 'Peak2Peak', 'Entropy', 'NumberOfPeaks',
       'SpecEntropy']
# sel_features_calc = ['MeanFreq', 'Amplitude', 'AUC', 'Duration',
#        'Peak2Peak', 'Power', 'Entropy', 'NumberOfPeaks', 'SpecEntropy']
for fold in folds.keys():
    df_train = df[~df.rat_id.isin(folds[fold])].sort_values(by=['rat_id', 'id'])#.drop(columns=['Peak2Peak', 'Power', 'NumberOfPeaks','Amplitude','AUC'])
    df_test = df[df.rat_id.isin(folds[fold])].sort_values(by=['rat_id', 'id'])#.drop(columns=['Peak2Peak', 'Power', 'NumberOfPeaks','Amplitude','AUC'])
    X = df_train[sel_features_calc].values
    y = df_train['label'].values.astype(int)

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # scaler = StandardScaler()   
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)

    #train a random forest classifier
    clf = RandomForestClassifier(n_estimators=500, max_depth=8,random_state=0)
    # clf = SVC(kernel='rbf', C=10, gamma=0.1)
    # clf = MLPClassifier(hidden_layer_sizes=(300,), max_iter=1000, alpha=0.0001, activation= 'tanh',learning_rate= 'constant',solver= 'adam',random_state=1)
    clf.fit(X_train, y_train)

    #validate the classifier
    y_pred = clf.predict(X_val)
    # classification report
    print('val',classification_report(y_val, y_pred))

    #plot feature importance
    start_time = time.time()
    result = permutation_importance(
        clf, X_val, y_val, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    feature_names = sel_features_calc
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    feat_importance = pd.concat([feat_importance,forest_importances],axis=1)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()
    
    #
    # #plot correlation
    #
    log_prob = clf.predict_log_proba(X_val)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    print(distance_matrix)
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
    cluster_ids = hierarchy.fcluster(dist_linkage, 0.4, criterion="distance")
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
        "Accuracy on test data with features removed: {:.2f}".format(
            clf_sel.score(X_val_sel, y_val)
        )
    )

    #
    ##plot feature importance
    #
    start_time = time.time()
    result = permutation_importance(
        clf_sel, X_val_sel, y_val, n_repeats=10, random_state=42, n_jobs=2
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    feature_names = df_train[sel_features_calc].columns[selected_features]
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    feat_importance = pd.concat([feat_importance,forest_importances],axis=1)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    ##%%
    # #test the classifier
    #
    # min_class_count = df_test.label.value_counts().min()
    # # get first n samples from each class
    # df_test = df_test.groupby('label').head(min_class_count)
    # X_test = df_test.drop(columns=['id','rat_id','label'])[sel_features_calc].values
    # y_test = df_test['label'].values.astype(int)
    # X_test = scaler.transform(X_test)
    # y_pred = clf.predict(X_test)    

    # print('test',classification_report(y_test, y_pred))
    # acc_values.append(accuracy_score(y_test, y_pred))
    # f1_values.append(f1_score(y_test, y_pred, average='macro'))  
    # #get confusion classes

    # cm = confusion_matrix(y_test, y_pred,normalize='true')
    # ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['complex_swr','swr','ripple']).plot()
    # plt.title('Random Forest')

    # plt.show()
    break

    # sns.heatmap(cm, annot=True, cmap='Blues',xticklabels=['complex_swr','swr','ripple'],yticklabels=['complex_swr','swr','ripple'])

