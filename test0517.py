import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import seaborn as sns
raw=pd.read_csv('u_protein.csv')
feature_names = raw.keys()
df = pd.DataFrame(raw[1::], columns=feature_names)
df['target'] = raw['protein']
df.dropna(axis=0, inplace=True)
df_copy = df
# print(df.info())
# print(df['target'].value_counts())
# plt.figure(figsize=(20,10))
# sns.heatmap(df.corr(),annot=True)
# plt.show()
x = df.iloc[:,1:22]
y = df['target']
for i in range(1,11):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33)
    # print(len(x_train),len(x_test),len(x_test)/len(x))
    # model = LogisticRegression(solver = 'liblinear')
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    model_pl = make_pipeline(StandardScaler(),LogisticRegression(solver = 'liblinear'))
    model_pl.fit(x_train, y_train)
    y_pred = model_pl.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm))
    train_acc = model_pl.score(x_train, y_train).round(2)
    test_acc = accuracy_score(y_test, y_pred).round(2)
    print(train_acc)
    print(test_acc)