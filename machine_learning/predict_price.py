import numpy as np  
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier()
data = np.array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
labels = np.array([1,1,1,2,2,2])
knn.fit(data,labels)
print knn.predict([[18,90]])  
