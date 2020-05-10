
# coding: utf-8

# ### Import packages
# Import necessry packages needed

# In[1]:


# import statements
from Planar_data_classification_lab_2_2 import * 
from Planar_data_classification_lab_2_3 import * 

pass


# ## Building the Nueral Network model ##
# 
# **Question**: Build your neural network model in `nn_model()`.
# 
# **Instructions**: The neural network model has to use the support functions we developed in *lab_2_2* and *lab_2_3* in the right order.

# In[2]:


# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
   
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
           
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        
    return parameters


# In[3]:


X_assess, Y_assess = nn_model_test_case()

parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
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
#     <td> [[-4.18494056  5.33220609]
#  [-7.52989382  1.24306181]
#  [-4.1929459   5.32632331]
#  [ 7.52983719 -1.24309422]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b1**</td>
#     <td> [[ 2.32926819]
#  [ 3.79458998]
#  [ 2.33002577]
#  [-3.79468846]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**W2**</td>
#     <td> [[-6033.83672146 -6008.12980822 -6033.10095287  6008.06637269]] </td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**b2**</td>
#     <td> [[-52.66607724]] </td> 
#   </tr>
#   
# </table>  

# ### Predictions
# 
# **Question**: Use your model to predict by building predict().
# Use forward propagation to predict results.
# 
# **Reminder**: predictions = $y_{prediction} = \mathbb 1 \text{{activation > 0.5}} = \begin{cases}
#       1 & \text{if}\ activation > 0.5 \\
#       0 & \text{otherwise}
#     \end{cases}$  
#     
# As an example, if you would like to set the entries of a matrix X to 0 and 1 based on a threshold you would do: ```X_new = (X > threshold)```
# 

# In[4]:



def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    
    A2, cache = forward_propagation(X, parameters)
    threshold = 0.5
    
    predictions = (A2 > threshold) 
        
    return predictions


# In[5]:


parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
if __name__=='__main__':
    print("predictions mean = " + str(np.mean(predictions)))


# **Expected Output**: 
# 
# 
# <table style="width:40%">
#   <tr>
#     <td>**predictions mean**</td>
#     <td> 0.666666666667 </td> 
#   </tr>
#   
# </table>
