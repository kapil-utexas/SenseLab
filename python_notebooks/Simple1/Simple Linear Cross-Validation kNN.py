import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.loadtxt("data.csv", delimiter=",")


# In[ ]:

print data


# In[ ]:

clf = KNeighborsClassifier(n_neighbors=5)


# In[ ]:

from sklearn import cross_validation


# In[ ]:

scores = cross_validation.cross_val_score(clf, data[:,:2], data[:,2], cv=5)


# In[ ]:

print scores


# In[ ]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:



