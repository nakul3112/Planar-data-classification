
# coding: utf-8

# In[1]:


# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model as slm
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1) # set a seed so that the results are consistent


# ## Dataset
# 
# First, let's get the dataset you will work on. The following code will load a "flower" 2-class dataset into variables `X` and `Y`.

# In[2]:


X, Y = load_planar_dataset()


# Visualize the dataset using matplotlib. The data looks like a "flower" with some red (label y=0) and some blue (y=1) points. Your goal is to build a model to fit this data. 

# In[3]:


# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral);


# You have:
#     - a numpy-array (matrix) X that contains your features (x1, x2)
#     - a numpy-array (vector) Y that contains your labels (red:0, blue:1).
# 
# Lets first get a better sense of what our data is like. 
# 
# **Exercise**: How many training examples do you have? In addition, what is the `shape` of the variables `X` and `Y`? 
# 
# **Hint**: How do you get the shape of a numpy array? [(help)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html)

# In[4]:


shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1] # training set size
if __name__=='__main__':
    print ('The shape of X is: ' + str(shape_X))
    print ('The shape of Y is: ' + str(shape_Y))
    print ('I have m = %d training examples!' % (m))


# **Expected Output**:
#        
# <table style="width:20%">
#   
#   <tr>
#     <td>**shape of X**</td>
#     <td> (2, 400) </td> 
#   </tr>
#   
#   <tr>
#     <td>**shape of Y**</td>
#     <td>(1, 400) </td> 
#   </tr>
#   
#     <tr>
#     <td>**m**</td>
#     <td> 400 </td> 
#   </tr>
#   
# </table>

# ## Simple Logistic Regression
# 
# Before building a full neural network, lets first see how logistic regression performs on this problem. You can use sklearn's built-in functions to do that. We can train a logisitc regression model by performing the following steps:
# - Use the following sklearn function to define the logistic regression model: *sklearn.linear_model.LogisticRegressionCV()*
# - Use X and Y to fit the model using the member function *fit(X,Y)* (remember to use transpose)

# In[5]:


# Train the logistic regression classifier
clf = slm.LogisticRegressionCV()
# fit the model using 
clf.fit(X.T, np.ravel(Y.T));


# You can now plot the decision boundary of these models. Run the code below.

# In[6]:


# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y.ravel())
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
if __name__=='__main__':
    print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
           '% ' + "(percentage of correctly labelled datapoints)")


# **Expected Output**:
# 
# <table style="width:20%">
#   <tr>
#     <td>**Accuracy**</td>
#     <td> 47% </td> 
#   </tr>
#   
# </table>
# 

# **Interpretation**: The dataset is not linearly separable, so logistic regression doesn't perform well. Hopefully a neural network will do better. Let's try this now! 

# Reference:
# - http://scs.ryerson.ca/~aharley/neural-networks/
# - http://cs231n.github.io/neural-networks-case-study/
