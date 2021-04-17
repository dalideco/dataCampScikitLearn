from sklearn import datasets
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

plt.style.use('ggplot')

iris = datasets.load_iris()
X=iris['data']
y=iris['target']

df = pd.DataFrame(X,columns=iris['feature_names'])
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X,y)

X_new = np.array([[2.0,1.4,0.2,1.5],[3.4,1.4,0.2,5.1],[0.2,7.5,3.5,3.4]])
prediction = knn.predict(X_new)


print(iris['target_names'])
print('prediction : {}'.format(prediction))

print(df.info())