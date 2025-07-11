import numpy as np
import random
"""
This is a very simple 2 --> 4 --> 1 neural net, built from scratch, to make predictions about the state of a XOR gate given inputs. It uses forward and backpropagation across two layers, 
the sigmoid function to normalize outputs, and 1000 iterations to converge to the correct output. 
"""

## Define smashing functions and their derivatives
def reLU(x):
    return np.maximum(0,x) # Locks activations to be >= 0
def reLU_derivative(x):
    return(x > 0).astype(float)
def sigmoid(x):
    return 1/(1+np.exp(-x)) # Exponentially squishes values to be between 0 and 1
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

learning_rate = 5 # How far to move per cycle

## Initialize input, output, and weights
X = np.array([[0,1],[1,0],[1,1],[0,0]])
y = np.array([[1,1,0,0]])

np.random.seed(1)

# Weights are normally distributed around zero
weights1 = np.random.randn(2,4)
bias1 = np.random.randn(1,4)

weights2 = np.random.randn(4,1)
bias2 = np.random.randn(1)

## Main training loop
for iter in range(1000) :
    # Layers themselves
    z1 = np.dot(X, weights1) + bias1 # Essentially a sum of the previous activations (In this case the inputs) times their weights, plus bias
    l1 = sigmoid(z1) # Smashes the outputs
    z2 = np.dot(l1, weights2) + bias2 
    l2 = sigmoid(z2)

    # Gets the distance from the real outputs
    error = l2.T - y # Shape is 4,1
    loss = np.mean(error ** 2) # Statistic for monitoring training
    # Backpropagation
    delta2 = error.T * sigmoid_deriv(z2) # distance from the reward times the weights of the last layer
    gradient_weights2 = np.dot(l1.T, delta2)/len(X) # chooses the right direction towards rewards
    gradient_bias2 = np.mean(delta2, axis=0) #           ^^^
    weights2 -= learning_rate * gradient_weights2 # moves "learning rate" towards reward
    bias2 -= learning_rate * gradient_bias2 # same as above, but with bias instead of weight
    # Same as below, but with the first layer
    delta1 = np.dot(delta2, weights2.T) * sigmoid_deriv(z1) # this line
    gradient_weights1 = np.dot(X.T, delta1)/len(X)
    gradient_bias1 = np.mean(delta1, axis=0)
    weights1 -= learning_rate * gradient_weights1
    bias1 -= learning_rate * gradient_bias1
    # Print shapes for debugging
    if iter == 0:
        print("X:", X.shape)
        print("z1:", z1.shape)
        print("a1:", l1.shape)
        print("z2:", z2.shape)
        print("a2:", l2.shape)
    # Print loss for training monitoring
    if iter % 100 == 0:
        print(f"loss at epoch {iter}: {loss}")

# Outputs
print("outputs are:", y.flatten())
print("predictions", (l2 > .5).astype(int).flatten())
print("raw is: ", l2.flatten())