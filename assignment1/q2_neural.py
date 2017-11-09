import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    # Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # YOUR CODE HERE: forward propagation
    # layer input
    N = data.shape[0]
    Z1 = np.dot(data, W1) + b1
    assert(Z1.shape == (N, H))
    A1 = sigmoid(Z1)
    # hidden layer
    Z2 = np.dot(A1, W2) + b2
    assert(Z2.shape == (N, Dy))
    # softmax
    A2 = softmax(Z2)
    # cost
    lost = labels * np.log(A2)
    cost = -np.sum(lost)
    # END YOUR CODE

    # YOUR CODE HERE: backward propagation
    delta2 = A2 - labels  # softmax error term, shape (N, Dy)
    gradW2 = np.dot(A1.T, delta2)  # shape (H, Dy)
    assert(gradW2.shape == W2.shape)
    gradb2 = np.sum(delta2, axis=0, keepdims=True)  # shape (1, Dy)
    assert(gradb2.shape == b2.shape)

    delta1 = np.dot(delta2, W2.T) * A1 * (1 - A1)  # shape (N, H)
    gradW1 = np.dot(data.T, delta1)  # shape(Dx, H)
    assert(gradW1.shape == W1.shape)
    gradb1 = np.sum(delta1, axis=0, keepdims=True)  # shape (1, H)
    assert(gradb1.shape == b1.shape)
    # END YOUR CODE

    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
                                                         dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    # YOUR CODE HERE

    # END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
