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



## Initialize input, output, and weights
X = np.array([[0,1],[1,0],[1,1],[0,0]])
y = np.array([[1], [1], [0], [0]])

np.random.seed(1)

learning_rate = .1 # How far to move per cycle
batch_size = 2 # How many input-output pairs to use each batch
n_batches = len(X)//batch_size
total_epochs = 10000

# Weights are normally distributed around zero
weights1 = np.random.randn(2,4)
bias1 = np.random.randn(1,4)

weights2 = np.random.randn(4,4)
bias2 = np.random.randn(4)

weights3 = np.random.randn(4,1)
bias3 = np.random.rand(1)

last_loss = 1

## Main training loop
for epoch in range(total_epochs):
    perm = np.random.permutation(len(X))
    X_shuffled = X[perm]
    y_shuffled = y[perm]


    epoch_loss = 0

    for i in range(n_batches) :

        start = i * batch_size
        end = start + batch_size

        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        # Layers themselves

        z1 = np.dot(X_batch, weights1) + bias1 # Essentially a sum of the previous activations (In this case the inputs) times their weights, plus bias
        l1 = reLU(z1) # Smashes the outputs

        z2 = np.dot(l1, weights2) + bias2 
        l2 = reLU(z2)

        z3 = np.dot(l2, weights3) + bias3
        l3 = sigmoid(z3)

        # Gets the distance from the real outputs

        error = l3 - y_batch # Shape is 4,1
        loss = np.mean(error ** 2) # Statistic for monitoring training
        epoch_loss += loss

        ## Backpropagation

        delta3 = error * sigmoid_deriv(z3)
        gradient_weights3 = np.dot(l2.T, delta3)/batch_size
        gradient_bias3 = np.mean(delta3, axis=0)
        weights3 -= learning_rate * gradient_weights3 # moves "learning rate" towards reward
        bias3 -= learning_rate * gradient_bias3

        delta2 = np.dot(delta3, weights3.T) * reLU_derivative(z2) # distance from the reward times the weights of the last layer
        gradient_weights2 = np.dot(l1.T, delta2)/batch_size # chooses the right direction towards rewards
        gradient_bias2 = np.mean(delta2, axis=0) #           ^^^
        weights2 -= learning_rate * gradient_weights2 
        bias2 -= learning_rate * gradient_bias2 # same as above, but with bias instead of weight

        # Same as above, but with the first layer
        delta1 = np.dot(delta2, weights2.T) * reLU_derivative(z1) # this line
        gradient_weights1 = np.dot(X_batch.T, delta1)/batch_size
        gradient_bias1 = np.mean(delta1, axis=0)
        weights1 -= learning_rate * gradient_weights1
        bias1 -= learning_rate * gradient_bias1

        # Print shapes for debugging
    if epoch == 0:
        print("X:", X.shape)
        print("z1:", z1.shape)
        print("a1:", l1.shape)
        print("z2:", z2.shape)
        print("a2:", l2.shape)
        print("z3:", z3.shape)
        print("a3:", l3.shape)
    # Print loss for training monitoring
    if epoch % 300 == 0:
        print(f"Loss at epoch {epoch}: {loss}")
        if (abs(last_loss-epoch_loss)) <=1e-4 :
            print(f'Early stopping at epoch {epoch}')
            break
        last_loss = epoch_loss

# Outputs
z1 = np.dot(X, weights1) + bias1
l1 = reLU(z1)

z2 = np.dot(l1, weights2) + bias2
l2 = reLU(z2)

z3 = np.dot(l2, weights3) + bias3
l3 = sigmoid(z3)

# Now l3 has shape (4, 1), matching y
print("outputs are:", y.flatten())
print("predictions", (l3 > .5).astype(int).flatten())
print("raw is: ", l3.flatten())