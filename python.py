import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris_dataset = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'])
iris_dataframe= pd.DataFrame(x_train,columns=iris_dataset.feature_names)
plot1=pd.plotting.scatter_matrix(iris_dataframe, c=y_train)
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(prediction)
print(iris_dataset['target_names'][prediction])
test_score=np.mean(prediction==y_test)
print(test_score)
