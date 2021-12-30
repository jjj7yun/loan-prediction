import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from collections import Counter

df = pd.read_csv('./archive/Training Data.csv')
df = df.drop("Id", axis=1)
df.head()

from sklearn.model_selection import train_test_split
train, valid_test = train_test_split(df.copy(), test_size=0.2, random_state=42)
valid, test = train_test_split(valid_test.copy(), test_size=0.5, random_state=42)


num_cols = [column for column in train.columns if train.dtypes[column] == "int64"]
cat_cols = [column for column in train.columns if train.dtypes[column] == "object"]

print("Numerical Columns : " + str(num_cols))
print("Categorical Columns : " + str(cat_cols))

train.hist(figsize=(12, 9))
plt.show()

corr_mat = train.corr()
sns.heatmap(corr_mat, vmin=-1, vmax=1, center=0, annot=True)
plt.plot()

fig = px.box(data_frame = train, y = 'Income', title = 'Box Plot for Income column')
fig.show()

fig = px.box(data_frame = train, y = 'Age', title = 'Box Plot for Age Column')
fig.show()


def print_count_cats(df, columns):
    for column in columns:
        count = len(df[column].value_counts())
        print("{0} : {1}".format(column, count))

print_count_cats(train, cat_cols)


train_data = df.copy()
train_data["House_Ownership"].replace({"rented": "Rented", "owned": "Owned", "norent_noown": "Neither Rent nor Own"}, inplace=True)
fig = go.Figure(data = [go.Pie(labels = train_data['House_Ownership'], pull=[0.05, 0])])
fig.update_traces(hoverinfo = 'label+percent', textinfo = 'label+value', textfont_size = 15, hole = 0.4,
                  marker = dict(line = dict(color = '#000000', width = 2)))
fig.update_layout( title_text = "House Ownership")
fig.show()


train_data = df.copy()
train_data["Car_Ownership"].replace({"no": "No", "yes": "Yes"}, inplace=True)
fig = go.Figure(data = [go.Pie(labels = train_data['Car_Ownership'])])
fig.update_traces(hoverinfo = 'label+percent', textinfo = 'label+value', textfont_size=15, hole = 0.4,
                  marker=dict(line = dict(color = '#000000', width = 2)))
fig.update_layout( title_text="Car Ownership")
fig.show()


train_data = df.copy()
train_data["Married/Single"].replace({"single": "Single", "married": "Married"}, inplace=True)
fig = go.Figure(data = [go.Pie(labels = train_data['Married/Single'])])
fig.update_traces(hoverinfo = 'label+percent', textinfo = 'label+value', textfont_size=15, hole = 0.4,
                  marker=dict(line = dict(color = '#000000', width = 2)))
fig.update_layout( title_text="Maritial Status")
fig.show()


train = train.drop(['CITY', 'Profession', 'STATE'], axis=1)
valid = valid.drop(['CITY', 'Profession', 'STATE'], axis=1)
test = test.drop(['CITY', 'Profession', 'STATE'], axis=1)


from sklearn.preprocessing import OrdinalEncoder

marital_enc = OrdinalEncoder(categories=[['single', 'married']])
train['Married/Single'] = marital_enc.fit_transform(train['Married/Single'].values.reshape(-1, 1))
valid['Married/Single'] = marital_enc.transform(valid['Married/Single'].values.reshape(-1, 1))
test['Married/Single'] = marital_enc.transform(test['Married/Single'].values.reshape(-1, 1))

house_enc = OrdinalEncoder(categories=[['norent_noown', 'rented', 'owned']])
train['House_Ownership'] = house_enc.fit_transform(train['House_Ownership'].values.reshape(-1, 1))
valid['House_Ownership'] = house_enc.transform(valid['House_Ownership'].values.reshape(-1, 1))
test['House_Ownership'] = house_enc.transform(test['House_Ownership'].values.reshape(-1, 1))

car_enc = OrdinalEncoder(categories=[['no', 'yes']])
train['Car_Ownership'] = car_enc.fit_transform(train['Car_Ownership'].values.reshape(-1, 1))
valid['Car_Ownership'] = car_enc.transform(valid['Car_Ownership'].values.reshape(-1, 1))
test['Car_Ownership'] = car_enc.transform(test['Car_Ownership'].values.reshape(-1, 1))

# prof_enc = OrdinalEncoder()
# train['Profession'] = prof_enc.fit_transform(train['Profession'].values.reshape(-1, 1))
# test['Profession'] = prof_enc.transform(test['Profession'].values.reshape(-1, 1))

# state_enc = OrdinalEncoder()
# train['STATE'] = prof_enc.fit_transform(train['STATE'].values.reshape(-1, 1))
# test['STATE'] = prof_enc.transform(test['STATE'].values.reshape(-1, 1))

print(train.shape)
print(valid.shape)
print(test.shape)


train_y = train['Risk_Flag'].copy()
train = train.drop('Risk_Flag', axis=1)

valid_y = valid['Risk_Flag'].copy()
valid = valid.drop('Risk_Flag', axis=1)

test_y = test['Risk_Flag'].copy()
test = test.drop('Risk_Flag', axis=1)


# 차트로 데이터 비율을 보여주는 함수
def showChart(dataOrigin):
    data = dataOrigin.copy()
    plt.subplots_adjust(left=1,right=3,bottom=1,top=2,wspace=0.2,hspace=0.4)
    plt.subplot(1,2,1)
    plt.title('Risk_Flag')
    plt.pie(data.groupby(data['Risk_Flag']).size(), labels=[0, 1])
    
    plt.subplot(1,2,2)
    sns.countplot(x='Risk_Flag', hue='Risk_Flag', data=data)
    
    plt.show()
    
    
    data_org = train.copy()
    
data_org['Risk_Flag'] = train_y
print("Majority/Minority class: " + str(Counter(train_y)))

from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.2, random_state=42)
print("Before sampling: " + str(Counter(train_y)))
train_comb, train_comb_y = smote.fit_resample(train, train_y)
print("After sampling: " + str(Counter(train_comb_y)))

data_smote = train_comb.copy()
data_smote['Risk_Flag'] = train_comb_y
showChart(data_smote)

a=int(train_comb_y)

from imblearn.under_sampling import EditedNearestNeighbours 

enn = EditedNearestNeighbours(n_neighbors=20, n_jobs=-1)
print("Before sampling: " + str(Counter(train_comb_y)))
train_comb, train_comb_y = enn.fit_resample(train_comb, train_comb_y)
print("After sampling: " + str(Counter(train_comb_y)))

data_enn = train_comb.copy()
data_enn['Risk_Flag'] = train_comb_y
showChart(data_enn)

#정규화

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_X = scaler.fit_transform(train)
valid_X = scaler.transform(valid)
test_X = scaler.transform(test)
train_comb_X = scaler.fit_transform(train_comb)


#model 생성
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_roc_curve

def metrics(model, valid_X, valid_y, pred):
    print(classification_report(valid_y, pred))
    print('roc_auc_score:', roc_auc_score(valid_y, pred))
    plot_confusion_matrix(model, valid_X, valid_y, values_format='d', display_labels=['0','1'])
    
    
    
from xgboost import XGBClassifier

params = {'learning_rate': 0.22657857685769822, 
                       'reg_lambda': 2.84652617858594444e-08, 
                       'reg_alpha': 1.31631262255538e-05, 
                       'subsample': 0.8648713185810271, 
                       'colsample_bytree': 0.2872653738124343, 
                       'max_depth': 4, 
                       'n_estimators': 1371}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, tree_method='gpu_hist', **params)
xgb.fit(train_X, train_y, eval_metric='logloss')
y_pred = xgb.predict(valid_X)

metrics(xgb, valid_X, valid_y, y_pred)


import optuna

def objective(trial):
    # setting search space
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)
    n_estimators = trial.suggest_int("n_estimators", 10, 5000)
    
    # defining the model
    clf = XGBClassifier(learning_rate=learning_rate, 
                        reg_lambda=reg_lambda,
                        subsample=subsample, 
                        colsample_bytree=colsample_bytree, 
                        tree_method='gpu_hist', predictor="gpu_predictor", # using gpu to speed up the process
                        max_depth=max_depth, 
                        n_estimators=n_estimators, 
                        use_label_encoder=False,
                        random_state=42)
    
    clf.fit(train_comb_X, train_comb_y, eval_metric='logloss')
    valid_preds = clf.predict(valid_X)
    score = roc_auc_score(valid_y, valid_preds)
    
    return score

# Hyperparameter 튜닝에 사용
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Print the result
best_params = study.best_params
best_score = study.best_value
print(f"Best score: {best_score}\n")
print(f"Optimized parameters: {best_params}\n")

best_param = {
    'learning_rate': 0.07721814588865467, 
    'reg_lambda': 4.577137308147702, 
    'reg_alpha': 1.10355169164888156, 
    'subsample': 0.1202409892812214, 
    'colsample_bytree': 0.9901267967538773, 
    'max_depth': 7, 
    'n_estimators': 1058
}
xgb_comb = XGBClassifier(random_state=42, use_label_encoder=False, tree_method='gpu_hist', **best_param)
xgb_comb.fit(train_comb_X, train_comb_y, eval_metric='logloss')
y_pred_comb = xgb_comb.predict(valid_X)

metrics(xgb_comb, valid_X, valid_y, y_pred_comb)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 400, criterion = 'entropy')
rf.fit(train_X, train_y)
y_pred = rf.predict(valid_X)

metrics(rf, valid_X, valid_y, y_pred)



from imblearn.under_sampling import EditedNearestNeighbours 

enn = EditedNearestNeighbours(n_neighbors=20, n_jobs=-1)
print("Before sampling: " + str(Counter(train_y)))
train_enn, train_enn_y = enn.fit_resample(train, train_y)
print("After sampling: " + str(Counter(train_enn_y)))

data_enn = train_comb.copy()
data_enn['Risk_Flag'] = train_enn_y
showChart(data_enn)

train_enn_X = scaler.fit_transform(train_enn)


from sklearn.ensemble import RandomForestClassifier

rf_enn = RandomForestClassifier(n_estimators = 400, criterion = 'entropy')
rf_enn.fit(train_enn_X, train_enn_y)
y_pred_enn = rf.predict(valid_X)

metrics(rf_enn, valid_X, valid_y, y_pred_enn)
showChart(data_org)
