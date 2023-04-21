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


#HPCbelo_features
#HPCpyra_features
def create_dataset(file_path='data/VEH_FEATURES_FINAL',feature_folder='HPCpyra_features'):
    label_dct = {'complex_swr':0,'swr':1,'ripple':2}
    label_dct = {'complex_swr':0,'swr':1,'sw':2}
    data_type = 'veh' if 'VEH' in file_path else 'cbd'    
    df = pd.DataFrame(columns=['id','rat_id','label','MeanFreq','Amplitude','AUC','Duration','Peak2Peak','Power','Entropy','NumberOfPeaks','SpecEntropy'])
    #walk through files in folder
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith('.mat'):
                filename = os.path.join(root, file)
                print(filename)
                ratID = int(filename.split('_')[-2].split('ratID')[-1])
                mat = loadmat(filename)

                for k in label_dct.keys():
                    data = mat['GC_' + k + '_ratID'+ str(ratID) +'_' + data_type][feature_folder][0,0]
                    df_tmp = pd.DataFrame(data,columns=['MeanFreq','Amplitude','AUC','Duration','Peak2Peak','Power','Entropy','NumberOfPeaks','SpecEntropy'])
                    df_tmp['id'] = df_tmp.index
                    df_tmp['rat_id'] = ratID    
                    df_tmp['label'] = label_dct[k]
                    df = pd.concat([df,df_tmp])
    df.to_csv('proc_data/veh_features_HPCpyra_final.csv',index=False)
# create_dataset(file_path='data/VEH_FEATURES_FINAL_2')
#%%
df = pd.read_csv('/home/ricardo/Documents/Projects/ripple-classifier/proc_data/labeling_data_test_2/res_400_labeling_data.csv')
#drop csv_1 and csv_2
df = df.drop(['csv_1','csv_2'],axis=1)
df.to_csv('manual_labeling_400.csv',index=False)
#%%
#agreement_majority_threshold is true and describe
df[(df.agreement_majority_threshold == False) & (df.threshold_label == 'swr')][['majority_vote']].value_counts()

# %%
df = pd.read_csv('proc_data/veh_features_hpcbelo_final.csv')
# df = df[~df.rat_id.isin([10,204])]
acc_values = []
f1_values = []
#set seed
np.random.seed(42)
df.columns
#%%
df.shape
# %%
#veh_features_hpcbelo_final
#veh_features_HPCpyra_final
df = pd.read_csv('proc_data/veh_features_hpcbelo_final.csv')
df = df[~df.rat_id.isin([10,204])]
df_belo = pd.read_csv('proc_data/veh_features_HPCpyra_final.csv')
df_belo = df_belo[~df_belo.rat_id.isin([10,204])]
#merge dataframes by id and rat_id
df = pd.merge(df,df_belo,on=['id','rat_id','label'],suffixes=('_hpc','_hpcbelo'))
acc_values = []
f1_values = []
#set seed
np.random.seed(42)
df.columns
sel_features_calc = ['MeanFreq_hpc', 'Amplitude_hpc', 'AUC_hpc',
       'Duration_hpc', 'Peak2Peak_hpc', 'Power_hpc', 'Entropy_hpc',
       'NumberOfPeaks_hpc', 'SpecEntropy_hpc', 'MeanFreq_hpcbelo',
       'Amplitude_hpcbelo', 'AUC_hpcbelo', 'Duration_hpcbelo',
       'Peak2Peak_hpcbelo', 'Power_hpcbelo', 'Entropy_hpcbelo',
       'NumberOfPeaks_hpcbelo', 'SpecEntropy_hpcbelo']
#%%
#veh
folds = {1: [206, 210], 2:[203,211],3:[9,201],4:[213,4,210]}
sel_features_calc = ['MeanFreq', 'AUC', 'Peak2Peak', 'Entropy', 'NumberOfPeaks',
       'SpecEntropy']
#cbd
# folds = {1: [214, 205], 2:[5,2],3:[207,212],4:[11,209]}
#%%
feat_importance = pd.DataFrame()
log_prob = None
df = pd.read_csv('proc_data/veh_features_hpcbelo.csv')
df = df[~df.rat_id.isin([10,204])]
sel_features_calc = ['MeanFreq', 'AUC', 'Peak2Peak', 'Entropy', 'NumberOfPeaks',
       'SpecEntropy']
sel_features_calc = ['MeanFreq', 'Amplitude', 'AUC', 
       'Peak2Peak', 'Power', 'Entropy', 'NumberOfPeaks', 'SpecEntropy']

# sel_features_calc = ['MeanFreq_hpc', 'Amplitude_hpc', 'AUC_hpc',
#        'Duration_hpc', 'Peak2Peak_hpc', 'Power_hpc', 'Entropy_hpc',
#        'NumberOfPeaks_hpc', 'SpecEntropy_hpc', 'MeanFreq_hpcbelo',
#        'Amplitude_hpcbelo', 'AUC_hpcbelo', 'Duration_hpcbelo',
#        'Peak2Peak_hpcbelo', 'Power_hpcbelo', 'Entropy_hpcbelo',
#        'NumberOfPeaks_hpcbelo', 'SpecEntropy_hpcbelo']
for fold in folds.keys():
    df_train = df[~df.rat_id.isin(folds[fold])].sort_values(by=['rat_id', 'id'])#.drop(columns=['Peak2Peak', 'Power', 'NumberOfPeaks','Amplitude','AUC'])
    df_test = df[df.rat_id.isin(folds[fold])].sort_values(by=['rat_id', 'id'])#.drop(columns=['Peak2Peak', 'Power', 'NumberOfPeaks','Amplitude','AUC'])
    X = df_train[sel_features_calc].values
    y = df_train['label'].values.astype(int)

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # scaler = StandardScaler()   
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)

    #train a random forest classifier
    clf = RandomForestClassifier(n_estimators=500, max_depth=8,random_state=42)
    #train a svm classifier
    # clf = SVC(random_state=42,probability=True)
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
    cluster_ids = hierarchy.fcluster(dist_linkage, 0.8, criterion="distance")
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

    # #test the classifier
    
    min_class_count = df_test.label.value_counts().min()
    # get first n samples from each class
    df_test = df_test.groupby('label').head(min_class_count)
    X_test = df_test[sel_features_calc].values
    y_test = df_test['label'].values.astype(int)
    # X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)    

    print('test',classification_report(y_test, y_pred))
    acc_values.append(accuracy_score(y_test, y_pred))
    f1_values.append(f1_score(y_test, y_pred, average='macro'))  
    #get confusion classes

    cm = confusion_matrix(y_test, y_pred,normalize='true')
    ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['complex_swr','swr','sw']).plot()
    plt.title('Random Forest')

    plt.show()
    break

    # sns.heatmap(cm, annot=True, cmap='Blues',xticklabels=['complex_swr','swr','ripple'],yticklabels=['complex_swr','swr','ripple'])

#%%
from sklearn.model_selection import RandomizedSearchCV
fold = 1
df_train = df[~df.rat_id.isin(folds[fold])].sort_values(by=['rat_id', 'id'])#.drop(columns=['Peak2Peak', 'Power', 'NumberOfPeaks','Amplitude','AUC'])
df_test = df[df.rat_id.isin(folds[fold])].sort_values(by=['rat_id', 'id'])#.drop(columns=['Peak2Peak', 'Power', 'NumberOfPeaks','Amplitude','AUC'])
X = df_train[sel_features_calc].values
y = df_train['label'].values.astype(int)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

min_class_count = df_test.label.value_counts().min()
# get first n samples from each class
df_test = df_test.groupby('label').head(min_class_count)
X_test = df_test[sel_features_calc].values
y_test = df_test['label'].values.astype(int)

X_train = df[sel_features_calc].values
y_train = df['label'].values.astype(int)
#%%
import cleanlab
# cleanlab works with **any classifier**. Yup, you can use sklearn/PyTorch/TensorFlow/XGBoost/etc.
cl = cleanlab.classification.CleanLearning(RandomForestClassifier(n_estimators=500, max_depth=8,random_state=42))

# cleanlab finds data and label issues in **any dataset**... in ONE line of code!
label_issues = cl.find_label_issues(X_train, y_train,)
#%%
# label_final = label_issues.query('is_label_issue == True').sort_values(by=['label_quality'])
label_old = label_issues.query('is_label_issue == True').sort_values(by=['label_quality'])
#%%
#get label_old where label is in label_final
label_old.query('index not in @label_final.index')
#%%
df_all_errors = df.iloc[label_issues.query('is_label_issue == True').sort_values(by=['label_quality']).index]
# df_all_errors.to_csv('label_issues.csv',index=False)
df_belo_errors = pd.read_csv('label_issues_belo.csv')
df_t = pd.merge(df_all_errors,df_belo_errors,how='left',on=['rat_id','id','label'],indicator='exists')
label = 1
rat_id = 203
id = 434
df_all_errors.query('label == @label and rat_id == @rat_id and id == @id')
#%%
# cleanlab trains a robust version of your model that works more reliably with noisy data.
cl.fit(X_train, y_train)
#%%
# cleanlab estimates the predictions you would have gotten if you had trained with *no* label issues.
print(classification_report(y_test, cl.predict(X_test)))

#%%
clf = RandomForestClassifier(n_estimators=500, max_depth=8,random_state=42)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))

#%%
# A true data-centric AI package, cleanlab quantifies class-level issues and overall data quality, for any dataset.
cleanlab.dataset.health_summary(y_train, confident_joint=cl.confident_joint)
#%%
#find best parameters for random forest
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}
# Create a based model  
rf = RandomForestClassifier()
# Instantiate the random search model
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1) 
# Fit the random search model
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)
#evaluate the best model    
clf = RandomForestClassifier(**rf_random.best_params_)  
clf.fit(X_train, y_train)
print('train',classification_report(y_train, clf.predict(X_train))) 
print('val',classification_report(y_val, clf.predict(X_val)))
#%%
 
rf_random.best_params_