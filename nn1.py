# My first attempt of building a NN from scratch in python
# Based on the ex4 of the ML course
import pandas as pd
import numpy as np
import csv as csv
from scipy.optimize import minimize, check_grad, approx_fprime
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
import time, timeit
#from multiprocessing import Pool
#p = Pool()

###########################################################

def logloss_mc(y_true, y_prob, epsilon=1e-15):
  """ Multiclass logloss
  This function is not officially provided by Kaggle, so there is no
  guarantee for its correctness.
  """
  # normalize
  y_prob = y_prob / y_prob.sum(axis=1).reshape(-1, 1)
  y_prob = np.maximum(epsilon, y_prob)
  y_prob = np.minimum(1 - epsilon, y_prob)
  # get probabilities
  y = [y_prob[i, j] for (i, j) in enumerate(y_true)]
  ll = - np.mean(np.log(y))
  return ll

###########################################################

#load data
def load_train_data(train_size=0.8):
  df = pd.read_csv('train.csv')
  X = df.values.copy()
  np.random.shuffle(X)
  X_train, X_valid, y_train, y_valid = train_test_split(
        X[:, 1:-1], X[:, -1], train_size=train_size,
    )
  print " -- Loaded data (training set size = %.1f)" %train_size
  return (X_train.astype(float), X_valid.astype(float),
            y_train.astype(str), y_valid.astype(str))
          
def load_test_data():
  df = pd.read_csv('test.csv')
  X = df.values
  X_test, ids = X[:, 1:], X[:, 0]
  return X_test.astype(float), ids.astype(str)

###########################################################

# Sigmoid function
def sigmoid(z):
  g = 1./(1. + np.exp(-z))
  return g

# Sigmoid gradient
def sigmoidGrad(z):
  g = sigmoid(z)*(1. - sigmoid(z))
  return g

###########################################################

# Randomly initialize the weights of a layer with L_in
# incoming connections and L_out outgoing connections
# Note that W should be set to a matrix of size(L_out, 1 + L_in) as
# the column row of W handles the "bias" terms

def randInitializeWeights(L_in, L_out):
  epsilon_init = np.sqrt(6)/np.sqrt(L_in+L_out)
  W = np.random.rand(L_out, 1 + L_in)*2.*epsilon_init - epsilon_init
  return W

###########################################################

#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

def predict(Theta1, Theta2, X):

  m = np.size(X, 0)
  num_labels = np.size(Theta2, 0)

  p = np.zeros((m, 1))

  h1 = sigmoid(np.dot(np.append(np.ones((m,1)),X,1),Theta1.T))  #h1 = sigmoid([ones(m, 1) X] * Theta1');
  h2 = sigmoid(np.dot(np.append(np.ones((m,1)),h1,1),Theta2.T))  #h2 = sigmoid([ones(m, 1) h1] * Theta2');

  #[dummy, p] = max(h2, [], 2);
  #print "h2",h2.shape
  return h2

###########################################################

def forward_prop(X, Theta1, Theta2):

  m = np.size(X, 0)

  A1 = X
  A1 = np.append(np.ones((m,1)),A1,1)   #[ones(m, 1) A1]
  Z2 = np.dot(A1,Theta1.T)   #A1*Theta1';
  A2 = sigmoid(Z2)
  A2 = np.append(np.ones((m,1)),A2,1)   #[ones(m, 1) A2];
  Z3 = np.dot(A2,Theta2.T)   #A2*Theta2';
  A3 = sigmoid(Z3)   # this will be my h(x)
  
  return A1, Z2, A2, Z3, A3

###########################################################

#NNCOSTFUNCTION Implements the neural network cost function for a two layer
#neural network which performs classification
#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
#   X, y, lambda) computes the cost and gradient of the neural network. The
#   parameters for the neural network are "unrolled" into the vector
#   nn_params and need to be converted back into the weight matrices. 
# 
#   The returned parameter grad should be a "unrolled" vector of the
#   partial derivatives of the neural network.

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, reg_param):

  #start_cost = timeit.default_timer()

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network
#Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
#                 hidden_layer_size, (input_layer_size + 1));
#Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
#                 num_labels, (hidden_layer_size + 1));

  Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size + 1)], \
                    (hidden_layer_size, (input_layer_size + 1)), order='F')

  Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size + 1):], \
                    (num_labels, (hidden_layer_size + 1)), order='F')
  
  m = np.size(X, 0)
  # You need to return the following variables correctly 
  J = 0

  # Part 1: Feedforward the neural network and return the cost in the
  #         variable J. After implementing Part 1, you can verify that your
  #         cost function computation is correct by verifying the cost
  #         computed in ex4.m

  A3 = forward_prop(X, Theta1, Theta2)[4]

  eye_matrix = np.eye(num_labels)
  Y = eye_matrix[y.astype(int),:]
   
  J = (1./m)*np.sum( -Y*np.log(A3) - (1.-Y)*np.log(1.-A3) )  # unreg cost
    
  # Now implement regularization term
  J_reg1, J_reg2 = 0, 0

  Theta1_tmp = Theta1
  Theta1_tmp[:,0:1] = 0
  J_reg1 = np.sum(Theta1_tmp**2)

  Theta2_tmp = Theta2
  Theta2_tmp[:,0:1] = 0
  J_reg2 = np.sum(Theta2_tmp**2)

  J = J + (J_reg1 + J_reg2)*reg_param/(2*m)   # reg cost

  #end_cost = timeit.default_timer()
  #print "    Cost run time: %f" %(end_cost-start_cost)

  return J


###########################################################

# Part 2: Implement the backpropagation algorithm to compute the gradients
#         Theta1_grad and Theta2_grad. You should return the partial derivatives of
#         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
#         Theta2_grad, respectively. After implementing Part 2, you can check
#         that your implementation is correct by running checkNNGradients
#
#         Note: The vector y passed into the function is a vector of labels
#               containing values from 1..K. You need to map this vector into a 
#               binary vector of 1's and 0's to be used with the neural network
#               cost function.

def nnGradient(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, reg_param):

  #start_grad = timeit.default_timer()

# Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
# for our 2 layer neural network

  Theta1 = np.reshape(nn_params[0:hidden_layer_size*(input_layer_size + 1)], \
                    (hidden_layer_size, (input_layer_size + 1)), order='F')

  Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size + 1):], \
                    (num_labels, (hidden_layer_size + 1)), order='F')
  
  m = np.size(X, 0)
  Theta1_grad = np.zeros(np.shape(Theta1))
  Theta2_grad = np.zeros(np.shape(Theta2))

  eye_matrix = np.eye(num_labels)
  Y = eye_matrix[y.astype(int),:]

  D_1 = np.zeros((hidden_layer_size, input_layer_size+1))
  D_2 = np.zeros((num_labels, hidden_layer_size+1))

  D_1x, D_2x = D_1, D_2

  A1, Z2, A2, Z3, A3 = forward_prop(X, Theta1, Theta2)

  d_3x = A3 - Y 
  Z2_prime = np.append(np.ones((m,1)),sigmoidGrad(Z2),1)
  d_2x = np.dot(Theta2.T,d_3x.T)*Z2_prime.T
  
  D_2x = np.dot(d_3x.T,A2)
  D_1x = np.dot(d_2x[1:,:],A1)

#  for i in range(m):
#    # forward propagation 
#    a_1 = np.append(1,X[i:i+1,:].T)   #a_1 = [1; X(i,:)'];
#    a_1 = np.reshape(a_1,(a_1.size,1))
#    z_2 = np.dot(Theta1,a_1)
#    a_2 = np.append(1,sigmoid(z_2))
#    a_2 = np.reshape(a_2,(a_2.size,1))
#    z_3 = np.dot(Theta2,a_2)
#    a_3 = sigmoid(z_3)
# 
#    # back propagation   
#    d_3 = a_3 - Y[i:i+1,:].T   #d_3 = a_3 - Y(i,:)';
#    d_2 = np.dot(Theta2.T,d_3)*np.vstack([1,sigmoidGrad(z_2)])
# 
#    #d_3x = A3[i:i+1,:] - Y[i:i+1,:]
#    #print d_3.shape, d_3x.shape
#    #print np.sum((d_3-d_3x))
# 
#    D_2 = D_2 + np.dot(d_3,a_2.T)
#    D_1 = D_1 + np.dot(d_2[1:],a_1.T)

  Theta1_grad = D_1x/m
  Theta2_grad = D_2x/m
  
# Part 3: Implement regularization with the cost function and gradients.

  #Theta1_tmp = Theta1
  #Theta2_tmp = Theta2
  #Theta1_tmp[:,0] = 0
  #Theta2_tmp[:,0] = 0

  Theta1_grad[:,1:] = Theta1_grad[:,1:] + (reg_param/m)*Theta1[:,1:]#_tmp
  Theta2_grad[:,1:] = Theta2_grad[:,1:] + (reg_param/m)*Theta2[:,1:]#_tmp
  
  Theta_grad_unrolled = np.append(Theta1_grad.T, Theta2_grad.T)
  
  #end_grad = timeit.default_timer()
  #print "Gradient run time: %f" %(end_grad-start_grad)
  return Theta_grad_unrolled

###########################################################

def main():

# Setup the parameters you will use for this exercise
  input_layer_size  = 93   # 93 features # 20x20 Input Images of Digits
  hidden_layer_size = 50 #100 #25   # 25 hidden units
  num_labels = 9           # 10 labels, from 1 to 10  

  print " - Begin!"
  print " -- Job started at %s" %time.strftime('%X')
  X_train, X_valid, y_train, y_valid = load_train_data(0.8)
  X_test, ids = load_test_data()
  
  m, n = np.shape(X_train)

  # Feature normalization
  X_mu = np.mean(X_train, axis=0)
  X_sigma = np.std(X_train, axis=0)
  X_train = (X_train - X_mu)/X_sigma
  X_valid = (X_valid - X_mu)/X_sigma
  X_test = (X_test - X_mu)/X_sigma

  # Recode labels
  encoder = LabelEncoder()
  y_train = encoder.fit_transform(y_train)
  y_valid = encoder.fit_transform(y_valid)

  print(" -- Initializing Neural Network Parameters...")
  initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
  initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
  #Unroll parameters, transpose to conform to matlab style
  initial_nn_params = np.append(initial_Theta1.T, initial_Theta2.T)

  ###################  More Parameters:
  reg_param = 1.3   # regularization lambda
  max_iters = 300 #500   # for the optimizer
  ########################################
  
  print "  Hidden Layer Size: %d" %hidden_layer_size
  print "  Lambda (reg): %.1f" %reg_param
  print "  Max Iterations: %d" %max_iters
  print " -- "
#  cost, grad = nnCostFunction(initial_nn_params, input_layer_size, \
#                 hidden_layer_size, num_labels, X_train, y_train, reg_param)
#  print "Cost,Grad",cost,grad

###########################################################
  def cost_train(nn_params):
    return nnCostFunction(nn_params, input_layer_size, \
          hidden_layer_size, num_labels, X_train, y_train, reg_param)
          
  def cost_grad(nn_params):
    return nnGradient(nn_params, input_layer_size, \
          hidden_layer_size, num_labels, X_train, y_train, reg_param)
###########################################################
  #print(" -- Checking Gradient...")
  #epsilon = 1.e-08
  #mygrad = cost_grad(initial_nn_params)
  #numgrad = approx_fprime(initial_nn_params, cost_train, 1e-04)

#  for i,j in zip(mygrad,numgrad):
#    print i,j
  #print numgrad

  #grad_check = np.sqrt(np.sum((mygrad - numgrad)**2))
  #print "Error:",grad_check    
  #print cost_train(initial_nn_params)       
  #print "Error:",check_grad(cost_train, cost_grad, initial_nn_params)
  #raise SystemExit()

  print(" -- Training Neural Network...")
  start = time.clock()

  options = {'maxiter': max_iters, 'disp': True}
  method = 'Newton-CG'#'BFGS'#'CG'
  
  OptSolution = minimize(cost_train,initial_nn_params,jac=cost_grad,options=options, method=method)
  best_nn_params = OptSolution.x
  
  end = time.clock()
  print " -- Elapsed time approx: %.2f" %(end - start)

  #best_nn_params = initial_nn_params

  #unroll parameters
  Theta1 = np.reshape(best_nn_params[0:hidden_layer_size*(input_layer_size + 1)], \
                    (hidden_layer_size, (input_layer_size + 1)), order='F')

  Theta2 = np.reshape(best_nn_params[hidden_layer_size*(input_layer_size + 1):], \
                    (num_labels, (hidden_layer_size + 1)), order='F')


  print " ** Training Set Cost: %.4f" %nnCostFunction(best_nn_params, input_layer_size, \
          hidden_layer_size, num_labels, X_train, y_train, reg_param)
  print " ** Validation Set Cost: %.4f" %nnCostFunction(best_nn_params, input_layer_size, \
          hidden_layer_size, num_labels, X_valid, y_valid, reg_param)

  h_train = predict(Theta1, Theta2, X_train)
  pred_train = np.argmax(h_train,1)
  print " ** Training Set Accuracy: %.2f" \
     %(np.mean(np.double(pred_train == y_train))*100)
  
  h_valid = predict(Theta1, Theta2, X_valid)
  pred_valid = np.argmax(h_valid,1)
  print " ** Validation Set Accuracy: %.2f" \
     %(np.mean(np.double(pred_valid == y_valid))*100)

  score = logloss_mc(y_valid, h_valid)
  print(" -- Multiclass logloss on validation set: {:.4f}.".format(score))

###########################################################
  #Make submission
  h_test = predict(Theta1, Theta2, X_test)
  h_test = np.round(h_test,3)
  
  with open("ottonn.csv", 'w') as f:
    f.write('id,')
    f.write(','.join(encoder.classes_))
    f.write('\n')
    for id, probs in zip(ids, h_test):
      probas = ','.join([id] + list(map(str, probs.tolist())))
      f.write(probas)
      f.write('\n')
    #print(" -- Wrote submission to file {}.".format(path))

  #np.savetxt("ottonn.csv",h_train,fmt="%.3f",delimiter=",")

  #predictions_file = open("ottonn.csv", "wb")
  #open_file_object = csv.writer(predictions_file)
  #open_file_object.writerow(["id","Class_1","Class_2","Class_3",\
  # "Class_4","Class_5","Class_6","Class_7","Class_8","Class_9"])
  #open_file_object.writerows(zip(ids, output_ok))

  #open_file_object.writerows(h_train)
  #predictions_file.close()

  print(" - Finished.")

if __name__ == '__main__':
    main()
