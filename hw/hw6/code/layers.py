from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    row_n = x.shape[0] # N
    col_n = np.prod(x.shape[1:]) # D
    x_0 = x.reshape(row_n, col_n) # (N * D)
    out = x_0 @ w + b # (N * D) @ (D * M) + (M,) = (N * M)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    row = x.shape[0] # N
    col = np.prod(x.shape[1:]) # D
    x1 = x.reshape(row, col) # (N * D)
    
    dx = (dout @ w.T).reshape(x.shape) # (N * M) @ (M * D) = (N * D) => (N, d_1, ... d_k)
    dw = x1.T @ dout # (D * N) @ (N * M) = (D * M)
    db = np.sum(dout, axis = 0) #dout.T @ np.ones(row) # (M * N) @ (N,) = (M,)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db
  

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    
#     dx = dout.T @ (x > 0)
    dx = (x > 0) * dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    loss = 0.0
    dx = None
    ###########################################################################
    # TODO: Implement the softmax loss                                        #
    ###########################################################################
#     row = x.shape[0]
#     Sc_m = x - np.max(x,axis = 1,keepdims = True)
#     e_Sc_m = np.exp(Sc_m)
#     E = -Sc_m[np.arange(row), y] + np.log(np.sum(e_Sc_m[np.arange(row), y], axis=1, keepdims=True))
#     loss = np.sum(E)
#     dx = (e_Sc_m / np.sum(e_Sc_m, axis=1, keepdims=True))[np.arange(row), y]- 1
    
    
    e_Sc_m = np.exp(x - np.max(x, axis=1, keepdims=True))
    E = e_Sc_m / np.sum(e_Sc_m, axis=1, keepdims=True)
    row = x.shape[0]
    loss = -np.sum(np.log(E[np.arange(row), y]))
    
    dx = E.copy()
    dx[np.arange(row), y] = E[np.arange(row), y] - 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx