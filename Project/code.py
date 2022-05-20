from google.colab import drive
drive.mount('/content/drive')



import pandas as pd

train_df = pd.read_csv ('/content/drive/MyDrive/train.csv')

display(train_df)


#NORMALIZING DATASET
import numpy as np
train_df = pd.DataFrame(np.random.randint(1,100, 50).reshape(-1, 1))
train_norm = train_df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
train_norm

#Libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_df=pd.read_csv('/content/drive/MyDrive/train.csv')
#CONVERTING STRING COLOUMN TO FLOAT
train_df = pd.DataFrame(train_df)
train_df['f_27'] = pd.to_numeric(train_df['f_27'], errors='coerce')
train_df = train_df.replace(np.nan, 0, regex=True)

y = train_df.target
X = train_df.drop('target', axis=1)

#dividing Data Into 80%(Train) And 20%(test)
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#Loading Training Data From Drive
test=pd.read_csv('/content/drive/MyDrive/test.csv')
test = pd.DataFrame(test)
test['f_27'] = pd.to_numeric(test['f_27'], errors='coerce')
test = test.replace(np.nan, 0, regex=True)


test.head()
#---------------------DECISION TREE CLASSIFIER--------------------------------------------
# # training a DescisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 2).fit(t_train, y_train)
# accuracy on t_test
accuracy1 = clf.score(t_test, y_test)
print ("ACCURACY DECISION TREE CLASSIFERS:", accuracy1)
clf.fit(abs(t_train),y_train)
target=clf.predict(test)
print("Predicted Values: ",target)

# #Exporting The Two Colomns(Id And target) Into exported colomn Csv
model_1_DTC = test[['id']].copy()
model_1_DTC['target'] = target
print(model_1_DTC)

"Creating Our Csv File With That Two Exported Columns For Submission On Kaggle"
model_1_DTC.to_csv('model_1_DTC.csv',index=False)

#----------------------NAIVE BAYES CLASSIFIER---------------------------------------------
# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(t_train, y_train)
# accuracy on t_test
accuracy2 = clf.score(t_test, y_test)
print ("ACCURACY OF NAIVE BAYES CLASSIFIERS:", accuracy2)
clf.fit(abs(t_train),y_train)
target = clf.predict(test)
print("Predicted Values: ",target)

# #Exporting The Two Colomns(Id And target) Into exported colomn Csv
model_2_NB = test[['id']].copy()
model_2_NB['target'] = target
print(model_2_NB)

"Creating Our Csv File With That Two Exported Columns For Submission On Kaggle"
model_2_NB.to_csv('model_2_NB.csv',index=False)

#----------------------KNN CLASSIFIER---------------------------------------------
# # training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(t_train, y_train)
# accuracy on t_test
accuracy3 = clf.score(t_test, y_test)
print ("ACCURACY OF KNN CLASSIFIERS:", accuracy3)
clf.fit(abs(t_train),y_train)
target=clf.predict(test)
print("Predicted Values: ",target)

#Exporting The Two Colomns(Id And target) Into exported colomn Csv
model_3_KNN = test[['id']].copy()
model_3_KNN['target'] = target
print(model_3_KNN)


"Creating Our Csv File With That Two Exported Columns For Submission On Kaggle"
model_3_KNN.to_csv('model_3_KNN.csv',index=False)
ACCURACY DECISION TREE CLASSIFERS: 0.5330888888888888
Predicted Values:  [0 0 0 ... 0 0 0]
             id  target
0        900000       0
1        900001       0
2        900002       0
3        900003       0
4        900004       0
...         ...     ...
699995  1599995       0
699996  1599996       0
699997  1599997       0
699998  1599998       0
699999  1599999       0

[700000 rows x 2 columns]
ACCURACY OF NAIVE BAYES CLASSIFIERS: 0.5523888888888889
Predicted Values:  [0 0 0 ... 0 1 1]
             id  target
0        900000       0
1        900001       0
2        900002       0
3        900003       1
4        900004       0
...         ...     ...
699995  1599995       0
699996  1599996       0
699997  1599997       0
699998  1599998       1
699999  1599999       1

[700000 rows x 2 columns]
ACCURACY OF KNN CLASSIFIERS: 0.5247444444444445
Predicted Values:  [0 0 0 ... 0 1 1]
             id  target
0        900000       0
1        900001       0
2        900002       0
3        900003       1
4        900004       0
...         ...     ...
699995  1599995       0
699996  1599996       0
699997  1599997       0
699998  1599998       1
699999  1599999       1

[700000 rows x 2 columns]
