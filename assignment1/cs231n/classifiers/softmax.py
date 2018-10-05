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
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    C = W.shape[1]
    N, D = X.shape

    d_helper = np.exp(X.dot(W))  # (N, C)
    # d_helper = X.dot(W)

    for i in range(N):
        scores = X[i].dot(W)
        scores -= np.max(scores)  # regularization
        # d_helper[i, :] = np.exp(d_helper[i, :])
        d_helper[i, :] /= np.sum(d_helper[i, :])
        divider = 0
        for j in range(C):
            # loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))
            divider += np.exp(scores[j])
            if j == y[i]:
                d_helper[i, j] -= 1  # -1, т.к. потом умножвем на X[i, j]
        loss -= np.log(np.exp(scores[y[i]]) / divider)

    dW = np.dot(X.T, d_helper)

    loss /= N
    dW /= N

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

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
    C = W.shape[1]
    N, D = X.shape

    d_helper = np.exp(X.dot(W))  # (N, C)

    scores = X.dot(W)  # (N, C)
    scores -= np.max(scores)  # dodge numeric instability
    output = np.exp(scores)
    output /= np.sum(output, axis=1, keepdims=True)

    loss -= np.log(output[np.arange(N), y])
    d_helper /= np.sum(d_helper, axis=1, keepdims=True)
    d_helper[np.arange(N), y] -= 1

    dW = np.dot(X.T, d_helper)

    loss = np.sum(loss) / N
    dW /= N

    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

