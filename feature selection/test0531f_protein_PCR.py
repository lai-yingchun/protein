import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score,recall_score,roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt 
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

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
# print(x.info())
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
x_train, y_train = SMOTE().fit_resample(x_train, y_train)
# x_train, y_train = TomekLinks().fit_resample(x_train, y_train)
model = XGBClassifier(
    eta=0.01,
    max_depth=3,
    n_estimators=100,
)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:,1]

scores=[]

for threshold in np.arange(0, 1, 0.1):
    y_pred = np.where(y_pred_proba >= threshold, 1, 0)
    prec = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    scores.append([threshold, prec, recall])
    df_pr = pd.DataFrame(scores, columns=['thresfold','precision', 'recall'])
    df_pr.sort_values(by='thresfold')
ax = df_pr.plot(x='precision', y='recall', marker='o')
ax.set_xlabel('category1 recall')
ax.set_ylabel('category1 precision')

for idx in df_pr.index:
    ax.text(x = df_pr.loc[idx, 'recall'],
    y = df_pr.loc[idx, 'precision']-0.02,
    s=df_pr.loc[idx, 'thresfold'].round(1))
plt.show()