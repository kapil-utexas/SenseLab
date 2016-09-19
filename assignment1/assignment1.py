import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Q1 _A:Find out what this relationship looks like by plotting a 2-D graph of the two features. You can use
#Python’s matplotlib for this task or any other plotting tool of your choice. Include the plot in your
#answer.

#plt.figure()
#plt.subplot(2, 1, 1)
data = pd.read_csv("q1_data_mod.csv")
features_to_plot = ['x','y','c']
get_ipython().magic(u'matplotlib ')
sns.pairplot(data, hue="c")
print("Close the opened plot to Open the Second Dataset Visualizer")
sns.plt.show(block=True)
#Q2_A : Plot the dataset q2_data.csv and indicate whether you think the two dataset classes are linearly
#separable. Include the plot in your answer.
data = pd.read_csv("q2_data_mod.csv")
features_to_plot = ['x','y','c']
get_ipython().magic(u'matplotlib ')
sns.pairplot(data, hue="c")
sns.plt.show(block=True)


