import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression #1
from sklearn.ensemble import RandomForestClassifier #2
from sklearn.ensemble import AdaBoostClassifier #3
from sklearn.tree import DecisionTreeClassifier #4
from sklearn.ensemble import GradientBoostingClassifier #5
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
raw=pd.read_csv('no_sleep.csv')
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

StandardScaler = StandardScaler()
x_num = StandardScaler.fit_transform(x_num)
cat_num = np.concatenate([x_cat,x_num], axis=1).round(2)
df1 = pd.DataFrame(cat_num, columns=new_key)

target_data=[]
for i,v in enumerate(raw.iloc[:,0]):
    target_data.append(v)
target = {'target':target_data}
df2 = pd.DataFrame(target)
df = pd.concat([df1,df2], axis=1)
    
# print(df['target'].value_counts())
# plt.figure(figsize=(20,10))
# sns.heatmap(df.corr(),annot=True)
# plt.show()

x = df.iloc[:,0:15]
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)
model_pl= Pipeline([('model',LogisticRegression())])
param_grid = [
    {'model' : [LogisticRegression()],
        'model__penalty':['l2'],
        'model__C':[0.001,0.01,1,5,10],
        'model__solver':['lbfgs','liblinear']},
    {'model' : [RandomForestClassifier()],
        'model__n_estimators':[50,150,200,250],
        'model__criterion':['gini','entropy'],
        'model__max_depth':[None,2,3,6],
        'model__min_samples_split':[2,6,8],
        'model__max_features':["auto"],
        'model__min_samples_leaf':[2,4,6,8]},
    {'model' : [AdaBoostClassifier()],
        'model__n_estimators':[50,150,200,250],
        'model__learning_rate':[1.0,1.5],
        'model__algorithm':['SAMME','SAMME.R']},
    {'model' : [DecisionTreeClassifier()],
        'model__criterion':['gini','entropy'],
        'model__splitter':['best','random'],
        'model__max_depth':[None,2,3,4,6],
        'model__min_samples_split':[2,4,6,8,20]},
    {'model' : [GradientBoostingClassifier()],
        'model__n_estimators':[50,150,200,250],
        'model__learning_rate':[1.0,1.5],
        'model__subsample':[0.5,0.8,1.0],
        'model__max_depth':[None,2,3,4,6],
        'model__min_samples_split':[2,4,6,8,20]},
]
cv_skf = StratifiedKFold(n_splits=10)
gs = GridSearchCV(model_pl, param_grid=param_grid, cv=cv_skf, return_train_score=True, verbose=1)
gs.fit(x_train,y_train)
score = gs.best_estimator_.score(x_test,y_test)
print()
print(gs.best_params_['model'])
print('訓練', gs.best_score_)
print('預測',score)


