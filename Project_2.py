# Load all needed libraries and modules:
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Kaggle specific import to load data set to kernel working space:
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Import CSV to Pandas DataFrame:
ddos_data = pd.read_csv("/kaggle/input/cse632-project2-training/training.csv", index_col=0)
# Number of rows = 760,426

# Alter rows having 'Infinity' or 'NaN' values for 'Flow Byts/s' and 'Flow Pkts/s':
ddos_data['Flow Byts/s'].fillna(0, inplace=True)
ddos_data.replace({np.inf:0}, inplace=True)

# Change 'Timestamp' data type:
ddos_data['Timestamp'] = pd.to_datetime(ddos_data['Timestamp'])

# Split data before preprocessing to prevent leakage:
## Set target data (y):
ddos_target = ddos_data.Label
## Set feature data (X):
ddos_data.drop(['Label'], axis=1, inplace=True)

# First split, create train and test. Results in 80/20 split:
ddos_train, ddos_test, ddos_train_target, ddos_test_target = train_test_split(ddos_data, ddos_target, test_size=0.2, random_state=27)

# Second split, create train and validation. Results in overall 60/20 split:
ddos_train, ddos_val, ddos_train_target, ddos_val_target = train_test_split(ddos_train, ddos_train_target, test_size=0.25, random_state=27)

# Drop 'Flow ID', 'Src IP', 'Dst IP' for all DataFrames:
ddos_Train = ddos_train.drop(['Flow ID', 'Src IP', 'Dst IP'], axis=1)
ddos_Val = ddos_val.drop(['Flow ID', 'Src IP', 'Dst IP'], axis=1)
ddos_Test = ddos_test.drop(['Flow ID', 'Src IP', 'Dst IP'], axis=1)

# Create new features from Timestamp feature:
def time_process(df):
    df['Time_Year'] = df['Timestamp'].dt.year
    df['Time_Month'] = df['Timestamp'].dt.month
    df['Time_Day'] = df['Timestamp'].dt.day
    df['Time_Hour'] = df['Timestamp'].dt.hour
    df['Time_Minute'] = df['Timestamp'].dt.minute
    df['Time_Second'] = df['Timestamp'].dt.second
    df['Time_Week'] = df['Timestamp'].dt.isocalendar()['week']
    df['Time_Weekday'] = df['Timestamp'].dt.isocalendar()['day']
    df.drop('Timestamp', axis=1, inplace=True)
    return df

time_process(ddos_Train)
time_process(ddos_Val)
time_process(ddos_Test)

# Label encode target data:
target_dict = {'Benign': 0, 'ddos': 1}
ddos_train_target = ddos_train_target.map(target_dict)
ddos_val_target = ddos_val_target.map(target_dict)
ddos_test_target = ddos_test_target.map(target_dict)

'''
(This code is commented out because it was used to conduct model tuning)
# Set scoring for GridSearchCV:
scoring = ['accuracy', 'f1', 'roc_auc']

# Set grid parameters for each model:
## LogisticRegression parameters:
lr_grid_params = {
    'penalty' : ['l2', 'l1'],
    'C' : np.logspace(-2, 4, 4),
    'solver' : ['liblinear', 'saga'],
    'max_iter' : [2500, 5000]
}

## RandomForestClassifier parameters:
rf_grid_params = {
    'n_estimators' : [100, 500],
    'criterion' : ['gini', 'entropy'],
    'max_features' : ['sqrt', 'log2'],
    'max_depth' : [50, 100, 150]    
}

## KNeighborsClassifier parameters:
nn_grid_params = {
    'n_neighbors' : [1, 5, 10],
    'weights' : ['uniform', 'distance'],
    'metric' : ['euclidean', 'minkowski']
}

# LogisticRegression grid search (This code is commented out because it was used to conduct model tuning):
lr_model = LogisticRegression()
lr_grid = GridSearchCV(lr_model, scoring=scoring, refit='f1', param_grid=lr_grid_params, verbose=1, cv=3, n_jobs=-1)
lr_grid.fit(ddos_Train, ddos_train_target)
lr_grid.best_estimator_
'''

# LogisticRegression model score check:
lr_model = LogisticRegression()
lr_model.fit(ddos_Train, ddos_train_target)
lr_pred = lr_model.predict(ddos_Val)
lr_model_acc = accuracy_score(ddos_val_target, lr_pred)
lr_model_f1 = f1_score(ddos_val_target, lr_pred)
lr_model_roc = roc_auc_score(ddos_val_target, lr_pred)
print("LR Model")
print("Accuracy:", lr_model_acc)
print("F1:", lr_model_f1)
print("ROC AUC:", lr_model_roc)

'''
# Random Forest grid search (This code is commented out because it was used to conduct model tuning):
rf_model = RandomForestClassifier()
rf_grid = GridSearchCV(rf_model, scoring=scoring, refit='f1', param_grid=rf_grid_params, verbose=1, cv=3, n_jobs=-1)
rf_grid.fit(ddos_Train, ddos_train_target)
rf_grid.best_estimator_
'''

# Random Forest model score check:
rf_model = RandomForestClassifier(criterion='entropy', max_depth=50, max_features='sqrt', n_estimators=500)
rf_model.fit(ddos_Train, ddos_train_target)
rf_pred = rf_model.predict(ddos_Val)
rf_model_acc = accuracy_score(ddos_val_target, rf_pred)
rf_model_f1 = f1_score(ddos_val_target, rf_pred)
rf_model_roc = roc_auc_score(ddos_val_target, rf_pred)
print("RF Model")
print("Accuracy:", rf_model_acc)
print("F1:", rf_model_f1)
print("ROC AUC:", rf_model_roc)

'''
(This code is commented out because it was used to conduct model tuning)
# K Nearest Neighbor grid search:
nn_model = KNeighborsClassifier()
nn_grid = GridSearchCV(nn_model, scoring=scoring, refit='f1', param_grid=nn_grid_params, verbose=1, cv=3, n_jobs=-1)
nn_grid.fit(ddos_Train, ddos_train_target)
nn_grid.best_estimator_
'''

# K Nearest Neighbor model scores check:
nn_model = KNeighborsClassifier(metric='euclidean', n_neighbors=1)
nn_model.fit(ddos_Train, ddos_train_target)
nn_pred = nn_model.predict(ddos_Val)
nn_model_acc = accuracy_score(ddos_val_target, nn_pred)
nn_model_f1 = f1_score(ddos_val_target, nn_pred)
nn_model_roc = roc_auc_score(ddos_val_target, nn_pred)
print("KNN Model")
print("Accuracy:", nn_model_acc)
print("F1:", nn_model_f1)
print("ROC AUC:", nn_model_roc)

# Run test data set predictions for each model:
lr_pred_final = lr_model.predict(ddos_Test)
rf_pred_final = rf_model.predict(ddos_Test)
nn_pred_final = nn_model.predict(ddos_Test)

# Create confusion matrices from test predictions:
lr_cm = confusion_matrix(ddos_test_target, lr_pred_final)
rf_cm = confusion_matrix(ddos_test_target, rf_pred_final)
nn_cm = confusion_matrix(ddos_test_target, nn_pred_final)

# Generate test accuracy scores:
lr_final_acc = accuracy_score(ddos_test_target, lr_pred_final)
rf_final_acc = accuracy_score(ddos_test_target, rf_pred_final)
nn_final_acc = accuracy_score(ddos_test_target, nn_pred_final)
print("LR Accuracy:", lr_final_acc)
print("RF Accuracy:", rf_final_acc)
print("KNN Accuracy:", nn_final_acc)

# Generate test F1 scores:
lr_final_f1 = f1_score(ddos_test_target, lr_pred_final)
rf_final_f1 = f1_score(ddos_test_target, rf_pred_final)
nn_final_f1 = f1_score(ddos_test_target, nn_pred_final)
print("LR F1:", lr_final_f1)
print("RF F1:", rf_final_f1)
print("KNN F1:", nn_final_f1)

# Create LR confusion matrix visualization:
categories = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
cat_counts = ["{}".format(val) for val in lr_cm.flatten()]
cat_percents = ["{:.2%}".format(val) for val in (lr_cm.flatten()/np.sum(lr_cm))]
labels = [f"{c1}\n{c2}\n{c3}" for c1, c2, c3 in zip(categories, cat_counts, cat_percents)]
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize=(10,10))
sns.set(font_scale=1.4)
res = sns.heatmap(lr_cm, annot=labels, fmt='', cmap='Blues', linewidths=0.1, linecolor='black')
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Classified Values\nAccuracy:{:.5f}\nF1 Score:{:.5f}'.format(lr_final_acc, lr_final_f1))
for _, spine in res.spines.items():
    spine.set_visible(True)
plt.show()

# Create RF confusion matrix visualizations:
categories = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
cat_counts = ["{}".format(val) for val in rf_cm.flatten()]
cat_percents = ["{:.2%}".format(val) for val in (rf_cm.flatten()/np.sum(rf_cm))]
labels = [f"{c1}\n{c2}\n{c3}" for c1, c2, c3 in zip(categories, cat_counts, cat_percents)]
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize=(10,10))
sns.set(font_scale=1.4)
res = sns.heatmap(rf_cm, annot=labels, fmt='', cmap='Greens', linewidths=0.1, linecolor='black')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Classified Values\nAccuracy:{:.5f}\nF1 Score:{:.5f}'.format(rf_final_acc, rf_final_f1))
for _, spine in res.spines.items():
    spine.set_visible(True)
plt.show()

# Create KNN confusion matrix visualizations:
categories = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
cat_counts = ["{}".format(val) for val in nn_cm.flatten()]
cat_percents = ["{:.2%}".format(val) for val in (nn_cm.flatten()/np.sum(nn_cm))]
labels = [f"{c1}\n{c2}\n{c3}" for c1, c2, c3 in zip(categories, cat_counts, cat_percents)]
labels = np.asarray(labels).reshape(2,2)
plt.figure(figsize=(10,10))
sns.set(font_scale=1.4)
res = sns.heatmap(nn_cm, annot=labels, fmt='', cmap='Reds', linewidths=0.1, linecolor='black')
plt.title('K-Nearest Neighbor Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Classified Values\nAccuracy:{:.5f}\nF1 Score:{:.5f}'.format(nn_final_acc, nn_final_f1))
for _, spine in res.spines.items():
    spine.set_visible(True)
plt.show()

# Generate ROC-AUC metrics:
lr_final_roc = roc_auc_score(ddos_test_target, lr_pred_final)
rf_final_roc = roc_auc_score(ddos_test_target, rf_pred_final)
nn_final_roc = roc_auc_score(ddos_test_target, nn_pred_final)
lr_fp, lr_tp, lr_thresh = roc_curve(ddos_test_target, lr_pred_final)
rf_fp, rf_tp, rf_thresh = roc_curve(ddos_test_target, rf_pred_final)
nn_fp, nn_tp, nn_thresh = roc_curve(ddos_test_target, nn_pred_final)

# Plot ROC curves:
plt.figure(figsize=(10,10))
plt.title('ROC Curve')
plt.plot(lr_fp, lr_tp, ls='dashed', color='blue', label="Logistic Regression: AUC = {:.5f}".format(lr_final_roc))
plt.plot(rf_fp, rf_tp, ls='dashdot',color='green', label="Random Forest: AUC = {:.5f}".format(rf_final_roc))
plt.plot(nn_fp, nn_tp, ls='dotted', color='red', label="K-Nearest Neighbor: AUC = {:.5f}".format(nn_final_roc))
plt.plot([0, 1], color='orange')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()

# Load competition testing data set to data frame:
ddos_comp = pd.read_csv("/kaggle/input/cse632-project-2-testing/testing.csv", index_col=0)

# Alter rows having 'Infinity' or 'NaN' values for 'Flow Byts/s' and 'Flow Pkts/s':
ddos_comp['Flow Byts/s'].fillna(0, inplace=True)
ddos_comp.replace({np.inf:0}, inplace=True)

# Change 'Timestamp' data type:
ddos_comp['Timestamp'] = pd.to_datetime(ddos_comp['Timestamp'])

# Drop 'Flow ID', 'Src IP', 'Dst IP' for the DataFrame:
ddos_Comp = ddos_comp.drop(['Flow ID', 'Src IP', 'Dst IP'], axis=1)

# Label encode time features for competition data set:
time_process(ddos_Comp)

# Use RandomForest model to classify competition data:
ddos_Comp_pred = rf_model.predict(ddos_Comp)

# Create DataFrame to store prediction results. A temporary DataFrame is used to transfer IDs (indices) to the final DataFrame:
df_temp = pd.DataFrame()
df_temp['TransactionID'] = ddos_Comp.index
comp_submission = pd.DataFrame(
    {
        'TransactionID' : df_temp['TransactionID'],
        'isFraud' : ddos_Comp_pred
    }
)

# Write competition classification results to CSV:
comp_submission.to_csv('Williams_Aaron_CSE632_Project2_Comp_Results.csv', index=False)
