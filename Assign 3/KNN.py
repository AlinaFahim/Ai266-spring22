import pandas as pd
train_df = pd.read_csv ('/content/drive/MyDrive/train.csv')
display(train_df)
import numpy as np
train_df = pd.DataFrame(np.random.randint(1,100, 50).reshape(-1, 1))
train_norm = train_df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
train_normal
import pandas as pd
from sklearn.model_selection import train_test_split
train_df=pd.read_csv('/content/drive/MyDrive/train.csv')
y = train_df.Cover_Type
X = train_df.drop('Cover_Type', axis=1)
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
test=pd.read_csv('/content/drive/MyDrive/test.csv')
test.head()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(t_train, y_train)
accuracy3 = clf.score(t_test, y_test)
print ("ACCURACY OF KNN :", accuracy3)
clf.fit(abs(t_train),y_train)
Cover_type=clf.predict(test)
print("Predicted Value: ",Cover_type)
