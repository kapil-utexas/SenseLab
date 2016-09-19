
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

dataset = np.loadtxt("q1_data.csv", delimiter=",")
#fig = plt.figure()
#plt.xlabel('Width')
#plt.ylabel('Height')
plt.close('all')

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(4, sharey=True)

axarr[0].plot(dataset[:,0],color = 'blue')
axarr[0].set_title('Feature Col 1')

axarr[1].plot(dataset[:,1],color = 'green')
axarr[1].set_title('Feature Col 2')

y = np.zeros(len(dataset))

col = 'r'
axarr[2].set_title('2D Feature Set')
axarr[2].scatter( dataset[:,0],y, color=col, s=1)

col = 'y'
axarr[3].scatter( dataset[:,1],y, color=col, s=1)
print(len(dataset[:,0]))
col = np.arange(25).reshape()
fig = plt.figure()
plt.scatter(dataset[:,0],dataset[:,1],color=col,s=1)
plt.xlabel('col1 vs col2')

plt.show(block=True)



