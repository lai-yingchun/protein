import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
import seaborn as sns
from imblearn.over_sampling import SMOTE

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
    
av = 0
x = df.iloc[:,0:15]
y = df['target']
for i in range(1,11):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)
    x_train, y_train = SMOTE().fit_resample(x_train, y_train)
    model = DecisionTreeClassifier(max_depth=6, random_state=42, min_samples_split=20)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm))
    print(classification_report(y_test, y_pred))
    train_acc = model.score(x_train, y_train).round(2)
    test_acc = accuracy_score(y_test, y_pred).round(2)
    av+=test_acc
    print(train_acc)
    print(test_acc)
av/=10
print(av)
