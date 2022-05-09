import pandas as panda
import numpy as np
import random

testDF = panda.read_csv('test.csv')

idDF = testDF[['id']]

idDF.insert(1,"target",0.0)

idDF['target'] = np.random.rand(700000,1)

idDF.to_csv('sample.csv')

print(idDF)
