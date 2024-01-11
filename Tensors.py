import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

### Defining a model using subclassing and specifying custom behavior ###

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class IdentityModel(tf.keras.Model):

  # In __init__ we define the Model's layers
  def __init__(self, n_output_nodes):
    super(IdentityModel, self).__init__()
    self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')

  def call(self, inputs, isidentity=False):
    x = self.dense_layer(inputs)
    if isidentity:
      return inputs 
    return x

  n_output_nodes = 3
  model = IdentityModel(n_output_nodes)

  x_input = tf.constant([[1,2.]], shape=(1,2))

  out_activate = model.call(x_input)
  out_identity = model.call(x_input, isidentity=True)

  print("Network output with activation: {}; network identity output: {}".format(out_activate.numpy(), out_identity.numpy()))

  ### Gradient computation with GradientTape ###

  # y = x^2
  # Example: x = 3.0
  x = tf.Variable(3.0)

  # Initiate the gradient tape
  with tf.GradientTape() as tape:
    # Define the function
    y = x * x
  # Access the gradient -- derivative of y with respect to x
  dy_dx = tape.gradient(y, x)

  ### Function minimization with automatic differentiation and SGD ###

  # Initialize a random value for our initial x
  x = tf.Variable([tf.random.normal([1])])
  print("Initializing x={}".format(x.numpy()))

  learning_rate = 1e-2 # learning rate for SGD
  history = []
  # Define the target value
  x_f = 4

  # We will run SGD for a number of iterations. At each iteration, we compute the loss,
  #   compute the derivative of the loss with respect to x, and perform the SGD update.
  for i in range(500):
    with tf.GradientTape() as tape:
      loss = (x - x_f)**2 # "forward pass": record the current loss on the tape

    # loss minimization using gradient tape
    grad = tape.gradient(loss, x) # compute the derivative of the loss with respect to x
    new_x = x - learning_rate*grad # sgd update
    x.assign(new_x) # update the value of x
    history.append(x.numpy()[0])

  # Plot the evolution of x as we optimize towards x_f!
  plt.plot(history)
  plt.plot([0, 500],[x_f,x_f])
  plt.legend(('Predicted', 'True'))
  plt.xlabel('Iteration')
  plt.ylabel('x value')