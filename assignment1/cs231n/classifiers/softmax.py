import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores = np.matmul(X[i],W)
    scores -= np.max(scores) # avoid numerical issue by making max score is 0
    scores = np.exp(scores)
    correct_class = y[i]
    loss -= np.log(scores[correct_class] / np.sum(scores))
    
    # Gradient
    for j in range(num_classes):
      if j == correct_class:
        dW[:,j] += -X[i].T + X[i].T*(scores[j]/np.sum(scores))
        continue
      dW[:,j] +=  X[i].T*(scores[j]/np.sum(scores)) 	
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  # Add regularization to the loss.
  loss /= num_train
  loss += reg * np.sum(W[0:-1] * W[0:-1])
  
  # Add regularization to the gradient
  dW /= num_train
  dW[0:-1] += 2*reg*W[0:-1] 
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

