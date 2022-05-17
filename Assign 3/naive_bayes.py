from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
train_df = pd.read_csv ('/content/drive/MyDrive/train.csv')
display(train_df)
#datasets
import numpy as np
train_df = pd.DataFrame(np.random.randint(1,100, 50).reshape(-1, 1))
train_norm = train_df.apply(lambda iterator: ((iterator.max() - iterator)/(iterator.max() - iterator.min())).round(2))
train_norm
#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

train_df=pd.read_csv('/content/drive/MyDrive/train.csv')

y = train_df.Cover_Type

X = train_df.drop('Cover_Type', axis=1)

#dividing into test and train datasets
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#connecting with cleaned dataset
test=pd.read_csv('/content/drive/MyDrive/test.csv')
test.head()
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB().fit(t_train, y_train)
# accuracy on t_test
accuracy2 = clf.score(t_test, y_test)
print ("ACCURACY OF NAIVE BAYES CLASSIFIERS:", accuracy2)
clf.fit(abs(t_train),y_train)
Cover_type = clf.predict(test)
print("Predicted Value: ",Cover_type)

model_2_NB = test[['Id']].copy()
model_2_NB['Cover_Type'] = Cover_type
print(model_2_NB)

"For kaggel submission now I make my file:"
model_2_NB.to_csv('model_2_NB.csv',index=False)
