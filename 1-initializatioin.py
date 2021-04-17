from sklearn import datasets #imports dataa from sickit-learn
import pandas as pd #library to structure data
import numpy as np #optimizes data structures
import matplotlib.pyplot as plt #better data visualization


plt.style.use('ggplot') #set the visualization

iris = datasets.load_iris() #loading datasets
type(iris) # shows that its a bunch
print(iris.keys()) #dict_keys(['data','target_names','DESCR','feature_names','target'])
type(iris['data']), type(iris['target']) #numpy.ndarray => shows that they are numyp arrays
print(iris['data'].shape) #(number of rows, number of colums)
print(iris['target_names']) #array([all the target types])



X = iris['data']
Y= iris['target']

df = pd.DataFrame(X, columns=iris['feature_names'])#building a data frame with panda
print(df.head()) #shows the first five rows
print(df.info()) #shows info about the columns (type and name) and memory usage
print(df.describe()) #shows statistics about the data


#in case of linear python
pd.plotting.scatter_matrix(df,c=Y,  figsize=[8,8],s=150,marker='D')#visualize data frame(s=markersize and marker=markershape)
plt.show()





