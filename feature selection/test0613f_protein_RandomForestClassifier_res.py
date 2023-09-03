import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score,recall_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import cross_val_score
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import TomekLinks
#資料前處裡
def classification_report_with_accuracy_score(y_true, y_pred):
    originalclass.extend(y_true)
    predictedclass.extend(y_pred)
    return accuracy_score(y_true, y_pred)

raw=pd.read_csv('f_protein.csv')
feature_names = raw.keys()
feature_names = feature_names[3::]
f_name=[]
for i, v in enumerate(feature_names):
    f_name.append(feature_names[i])
raw.dropna(axis=0, inplace=True)

x_cat = raw.iloc[:,1:3]
OneHotEncoder = OneHotEncoder(sparse=False)
x_cat = OneHotEncoder.fit_transform(x_cat)
key123 = OneHotEncoder.get_feature_names(['group','Exercise'])
new_key = np.concatenate([key123,f_name], axis=0)

x_num = raw.iloc[:,3::]

StandardScaler = MinMaxScaler()
x_num = StandardScaler.fit_transform(x_num)
cat_num = np.concatenate([x_cat,x_num], axis=1).round(2)
df1 = pd.DataFrame(cat_num, columns=new_key)

target_data=[]
for i,v in enumerate(raw.iloc[:,0]):
    target_data.append(v)
target = {'target':target_data}
df2 = pd.DataFrame(target)
df = pd.concat([df1,df2], axis=1)

x = df.iloc[:,0:19]
y = df['target']
print(x.info())
print(df['target'].value_counts())

#交叉驗證
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)
model = RandomForestClassifier(
    min_samples_leaf=2,
    n_estimators=50)
model.fit(x_train, y_train)
scores_acc = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=5)
scores_rc = cross_val_score(model, x_train, y_train, scoring='recall', cv=5)
print(f'5折交叉驗證精確率 : {np.mean(scores_acc)}')
print(f'5折交叉驗證召回率 : {np.mean(scores_rc)}')
#測試
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:,1]

# 排序特徵重要性
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
# # 打印特徵重要性排名
for f in range(x.shape[1]):
    print(f"{f + 1}. 特徵 '{new_key[indices[f]]}' 的重要性為 {importances[indices[f]]}")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm))
print(classification_report(y_test, y_pred))
test_acc = accuracy_score(y_test, y_pred).round(2)
print('正確率:', test_acc)

fpr, tpr, thres = roc_curve(y_test, y_pred_proba, pos_label=1)
df_roc = pd.DataFrame(zip(thres, fpr, tpr), columns=['threshold','1-specificity','sensitivity'])
# print(df_roc)
ax = df_roc.plot(x='1-specificity', y='sensitivity', marker='o')
for idx in df_roc.index:
    ax.text(x=df_roc.loc[idx,'1-specificity'], y=df_roc.loc[idx,'sensitivity']-0.05,s=df_roc.loc[idx,'threshold'].round(2))
plt.show()
auc = roc_auc_score(y_test, y_pred_proba)
print('AUC: ',auc)