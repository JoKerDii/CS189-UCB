from builtins import range
from builtins import object
import numpy as np

from layers import *


class FullyConnectedNet(object):
    """
    A fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of [H, ...], and perform classification over C classes.

    The architecure should be like affine - relu - affine - softmax for a one
    hidden layer network, and affine - relu - affine - relu- affine - softmax for
    a two hidden layer network, etc.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim, hidden_dim=[10, 5], num_classes=10,
                 weight_scale=0.1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: A list of integer giving the sizes of the hidden layers
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        """
        self.params = {}
        self.hidden_dim = hidden_dim
        self.n_layers = 1 + len(hidden_dim)
        ############################################################################
        # TODO: Initialize the weights and biases of the net. Weights              #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        ##### two layers
#         self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim[0]))
#         self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim[0], num_classes))
#         self.params['b1'] = np.zeros(hidden_dim[0])
#         self.params['b2'] = np.zeros(num_classes)
        ######
    
        ##### multi-layers
        for i in range(self.n_layers):
            Wi = 'W' + str(i+1)
            bi = 'b' + str(i+1)
            
            # First hidden layer
            if i == 0:
                self.params[Wi] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim[i]))              
                self.params[bi] = np.zeros(hidden_dim[i])
            # Output layer
            elif i == self.n_layers - 1:
                self.params[Wi] = np.random.normal(scale=weight_scale, size=(hidden_dim[i-1], num_classes))              
                self.params[bi] = np.zeros(num_classes)
            # Intermediate hidden layer   
            else:
                self.params[Wi] = np.random.normal(scale=weight_scale, size=(hidden_dim[i-1], hidden_dim[i])) 
                self.params[bi] = np.zeros(hidden_dim[i])
         ######
                
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        caches = {}
        ############################################################################
        # TODO: Implement the forward pass for the net, computing the              #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        ###### two layers
#         out1, cache1 = affine_forward(X, self.params['W1'], self.params['b1'])
#         out2, cache2 = relu_forward(out1)
        
#         out3, cache3 = affine_forward(out2, self.params['W2'], self.params['b2'])
#         scores = out3
        ######
    
        ###### multi-layers
        for i in range(self.n_layers-1):
            Wi = 'W' + str(i+1)
            bi = 'b' + str(i+1)
            
            # First hidden layer
            if i == 0:
                out = X
                
            # Intermediate layers
            out, caches['cache1 %d'%(i+1)] = affine_forward(out, self.params[Wi], self.params[bi])
            out, caches['cache2 %d'%(i+1)] = relu_forward(out)
            
            
        # The last layer
        scores, caches['cache3'] = affine_forward(out, self.params['W'+str(self.n_layers)],self.params['b'+str(self.n_layers)])
 
            

        ######
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the net. Store the loss            #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k].                                                          #
        ############################################################################
        
        loss, dscores = softmax_loss(scores, y)
        
        ###### two layers
#         dx2, grads['W2'], grads['b2'] = affine_backward(dscores, cache3)

#         dx1 = relu_backward(dx2, cache2)
#         dx0, grads['W1'], grads['b1'] = affine_backward(dx1, cache1)
        
#         grads['W2'] -= 1e-3 * self.params['W2'] # 1e-3 is good
#         grads['W1'] -= 1e-3 * self.params['W1']
        ######

        ###### multi-layers
        
        for i in range(self.n_layers, 0, -1):
            if i == self.n_layers:
                # hidden layer
                dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dscores, caches['cache3'])
            else:    
                dout = relu_backward(dout, caches['cache2 %d'%i])
                dout, grads['W'+str(i)], grads['b'+str(i)] = affine_backward(dout, caches['cache1 %d'%i])
            
            grads['W'+str(i)] -= 1e-3 * self.params['W'+str(i)]

        ######
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads 