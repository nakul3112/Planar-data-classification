
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


# ## Neural Network model
# 
# You are going to train a Neural Network with a single hidden layer.
# 
# **Here is our model**:
# <img src="images/classification_kiank.png" style="width:600px;height:300px;">
# 
# **Mathematically**:
# 
# For one example $x^{(i)}$:
# $$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1] (i)}\tag{1}$$ 
# $$a^{[1] (i)} = \tanh(z^{[1] (i)})\tag{2}$$
# $$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2] (i)}\tag{3}$$
# $$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$
# $$y^{(i)}_{prediction} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{5}$$
# 
# Given the predictions on all the examples, you can also compute the cost $J$ as follows: 
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$
# 
# **Reminder**: The general methodology to build a Neural Network is to:
#     1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
#     2. Initialize the model's parameters
#     3. Loop:
#         - Implement forward propagation
#         - Compute loss
#         - Implement backward propagation to get the gradients
#         - Update parameters (gradient descent)
# 
# You often build helper functions to compute steps 1-3 and then merge them into one function we call `nn_model()`. Once you've built `nn_model()` and learnt the right parameters, you can make predictions on new data.

# ### Defining the neural network structure ####
# 
# **Exercise**: Define three variables:
#     - n_x: the size of the input layer
#     - n_h: the size of the hidden layer (set this to 4) 
#     - n_y: the size of the output layer
# 
# **Hint**: Use shapes of X and Y to find n_x and n_y. Also, hard code the hidden layer size to be 4.

# In[2]:


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    
    n_x = X.shape[0] 
    n_h = 4
    n_y = Y.shape[0] 
    
    return (n_x, n_h, n_y)


# In[3]:


X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
if __name__=='__main__':
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))


# **Expected Output** (these are not the sizes you will use for your network, they are just used to assess the function you've just coded).
# 
# <table style="width:20%">
#   <tr>
#     <td>**n_x**</td>
#     <td> 5 </td> 
#   </tr>
#   
#     <tr>
#     <td>**n_h**</td>
#     <td> 4 </td> 
#   </tr>
#   
#     <tr>
#     <td>**n_y**</td>
#     <td> 2 </td> 
#   </tr>
#   
# </table>

# ### Initialize the model's parameters ####
# 
# **Exercise**: Implement the function `initialize_parameters()`.
# 
# **Instructions**:
# - Make sure your parameters' sizes are right. Refer to the neural network figure above if needed.
# - You will initialize the weights matrices with random values. 
#     - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
# - You will initialize the bias vectors as zeros. 
#     - Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.

# In[4]:


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) 
    
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros(shape=(n_h,1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros(shape=(n_y,1))
    
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[5]:


n_x, n_h, n_y = initialize_parameters_test_case()

parameters = initialize_parameters(n_x, n_h, n_y)

if __name__=='__main__':
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


# **Expected Output**:
# 
# <table style="width:90%">
#   <tr>
#     <td>**W1**</td>
#     <td> [[-0.00416758 -0.00056267]
#  [-0.02136196  0.01640271]
#  [-0.01793436 -0.00841747]
#  [ 0.00502881 -0.01245288]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**b1**</td>
#     <td> [[ 0.]
#  [ 0.]
#  [ 0.]
#  [ 0.]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**W2**</td>
#     <td> [[-0.01057952 -0.00909008  0.00551454  0.02292208]]</td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**b2**</td>
#     <td> [[ 0.]] </td> 
#   </tr>
#   
# </table>
# 
# 

# ### The Loop ####
# 
# **Question**: Implement `forward_propagation()`.
# 
# **Instructions**:
# - Look above at the mathematical representation of your classifier.
# - You can use the function `sigmoid()`. It is built-in (imported) in the notebook.
# - You can use the function `np.tanh()`. It is part of the numpy library.
# - The steps you have to implement are:
#     1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
#     2. Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all your predictions on all the examples in the training set).
# - Values needed in the backpropagation are stored in "`cache`". The `cache` will be given as an input to the backpropagation function.

# In[6]:


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    
    Z1 = np.matmul(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# In[7]:


X_assess, parameters = forward_propagation_test_case()

A2, cache = forward_propagation(X_assess, parameters)

if __name__=='__main__':
# Note: we use the mean here just to make sure that your output matches ours. 
    print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))


# **Expected Output**:
# <table style="width:55%">
#   <tr>
#     <td> -0.000499755777742 -0.000496963353232 0.000438187450959 0.500109546852 </td> 
#   </tr>
# </table>

# Now that you have computed $A^{[2]}$ (in the Python variable "`A2`"), which contains $a^{[2](i)}$ for every example, you can compute the cost function as follows:
# 
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small\tag{13}$$
# 
# **Exercise**: Implement `compute_cost()` to compute the value of the cost $J$.
# 
# **Instructions**:
# - There are many ways to implement the cross-entropy loss. To help you, we give you how we would have implemented
# $- \sum\limits_{i=0}^{m}  y^{(i)}\log(a^{[2](i)})$:
# ```python
# logprobs = np.multiply(np.log(A2),Y)
# cost = - np.sum(logprobs)                # no need to use a for loop!
# ```
# 
# (you can use either `np.multiply()` and then `np.sum()` or directly `np.dot()`).
# -  Use `np.squeeze()` to make sure cost dimension is a value rather than a array (turns [[17]] into 17 ). 
# 

# In[8]:


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example
    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2),1 - Y)
    cost = -np.sum(logprobs) / m
     # makes sure cost dimension is a floating point value.
    cost = np.squeeze(cost)
                                
    assert(isinstance(cost, float))
    
    return cost


# In[9]:


A2, Y_assess, parameters = compute_cost_test_case()

if __name__=='__main__':
    print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# **Expected Output**:
# <table style="width:20%">
#   <tr>
#     <td>**cost**</td>
#     <td> 0.692919893776 </td> 
#   </tr>
#   
# </table>
