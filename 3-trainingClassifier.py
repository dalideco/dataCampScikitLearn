from sklearn.neighbors import KNeighborsClassifier #inporting the classifier
knn = KNeighborsClassifier(n_neighbors=6)#set the neighbors
#requires: 
    # data numpy array or pandas DataFrame
    # features take on continuous values and non categories
    # no missing values in the data
    # each column is a feature and each row is an observation
    # target is a column with the same number of observations as the data


knn.fit(iris['data'], iris['target'])#fit it to our training set(give it the data and the target ) the program will learn
#returns the classifier itself and modifies to fit it to the data

#predicting
X_new =  np.array([
    [,,,,]
    ,[,,,,],
    [,,,,,]]) #gets the observations to predict

prediction = knn.predict(X_news)#predicts and returns an array containing the results
print('prediction :{}'.format(prediction))#prediction: [1 1 0]


#in the voting example
Y=df['party'].values
X=df.drop('party',axis=1).values #returns the values of the features without the target
knn.fit(X,Y)#fits the data