
# coding: utf-8

# In[1]:


# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1) # set a seed so that the results are consistent


# ## Backward Propagation
# 
# **Question**: Implement the function `backward_propagation()`.
# 
# **Instructions**:
# Backpropagation is usually the hardest (most mathematical) part in deep learning. To help you, here again is the slide from the lecture on backpropagation. You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.  
# 
# <img src="images/grad_summary.png" style="width:600px;height:300px;">
# 
# <!--
# $\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } = \frac{1}{m} (a^{[2](i)} - y^{(i)})$
# 
# $\frac{\partial \mathcal{J} }{ \partial W_2 } = \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } a^{[1] (i) T} $
# 
# $\frac{\partial \mathcal{J} }{ \partial b_2 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)}}}$
# 
# $\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} } =  W_2^T \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } * ( 1 - a^{[1] (i) 2}) $
# 
# $\frac{\partial \mathcal{J} }{ \partial W_1 } = \frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} }  X^T $
# 
# $\frac{\partial \mathcal{J} _i }{ \partial b_1 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)}}}$
# 
# - Note that $*$ denotes elementwise multiplication.
# - The notation you will use is common in deep learning coding:
#     - dW1 = $\frac{\partial \mathcal{J} }{ \partial W_1 }$
#     - db1 = $\frac{\partial \mathcal{J} }{ \partial b_1 }$
#     - dW2 = $\frac{\partial \mathcal{J} }{ \partial W_2 }$
#     - db2 = $\frac{\partial \mathcal{J} }{ \partial b_2 }$
#     
# !-->
# 
# - Tips:
#     - To compute dZ1 you'll need to compute $g^{[1]'}(Z^{[1]})$. Since $g^{[1]}(.)$ is the tanh activation function, if $a = g^{[1]}(z)$ then $g^{[1]'}(z) = 1-a^2$. So you can compute 
#     $g^{[1]'}(Z^{[1]})$ using `(1 - np.power(A1, 2))`.

# In[2]:


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = Y.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    
    W1 = parameters['W1']
    W2 = parameters['W2']
            
    # Retrieve also A1 and A2 from dictionary "cache".
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2= A2 - Y 
    dW2 = (1/m) * np.matmul(dZ2,A1.T) 
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims =True)
    dZ1 = np.matmul(W2.T,dZ2)*(1- A1**2)
    dW1 = (1/m) * np.matmul(dZ1,X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims = True)
    
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


# In[3]:


parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)

if __name__=='__main__':
    print ("dW1 = "+ str(grads["dW1"]))
    print ("db1 = "+ str(grads["db1"]))
    print ("dW2 = "+ str(grads["dW2"]))
    print ("db2 = "+ str(grads["db2"]))


# **Expected output**:
# 
# 
# 
# <table style="width:80%">
#   <tr>
#     <td>**dW1**</td>
#     <td> [[ 0.01018708 -0.00708701]
#  [ 0.00873447 -0.0060768 ]
#  [-0.00530847  0.00369379]
#  [-0.02206365  0.01535126]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**db1**</td>
#     <td>  [[-0.00069728]
#  [-0.00060606]
#  [ 0.000364  ]
#  [ 0.00151207]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**dW2**</td>
#     <td> [[ 0.00363613  0.03153604  0.01162914 -0.01318316]] </td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**db2**</td>
#     <td> [[ 0.06589489]] </td> 
#   </tr>
#   
# </table>  

# **Question**: Implement the update rule. Use gradient descent. You have to use (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2).
# 
# **General gradient descent rule**: $ \theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning rate and $\theta$ represents a parameter.
# 
# **Illustration**: The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.
# 
# <img src="images/sgd.gif" style="width:400;height:400;"> <img src="images/sgd_bad.gif" style="width:400;height:400;">
# 
# 

# In[4]:


def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
       
    # Retrieve each gradient from the dictionary "grads"
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    
    # Update rule for each parameter
   
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[5]:


parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

if __name__=='__main__':
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


# **Expected Output**:
# 
# 
# <table style="width:80%">
#   <tr>
#     <td>**W1**</td>
#     <td> [[-0.00643025  0.01936718]
#  [-0.02410458  0.03978052]
#  [-0.01653973 -0.02096177]
#  [ 0.01046864 -0.05990141]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b1**</td>
#     <td> [[ -1.02420756e-06]
#  [  1.27373948e-05]
#  [  8.32996807e-07]
#  [ -3.20136836e-06]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**W2**</td>
#     <td> [[-0.01041081 -0.04463285  0.01758031  0.04747113]] </td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**b2**</td>
#     <td> [[ 0.00010457]] </td> 
#   </tr>
#   
# </table>  
