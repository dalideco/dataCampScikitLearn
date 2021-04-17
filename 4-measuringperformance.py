from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()

X= iris['data']
y=iris['target']

df = pd.DataFrame(X, columns=iris['feature_names'])

print(df.head())



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=21, stratify=y)#randomly splits the data
# testsize : proportion of the original data is used
# random_state : set a seed for the random number generator 
#     that splits the data into train and test
# returns 4 arrays the training data , the test data ,the training labels and the test labels
# by default : 75% training data and 25% test data
# here we set it to 30 % using test_size
#stratify=y to make the labels distributed as they are in the original dataset

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)#training the test and seein the result
print('Test set predictions : \n {} \n real data test : \n {}'.format(y_pred,y_test))#comparing it to the real data target values
print(knn.score(X_test,y_test))