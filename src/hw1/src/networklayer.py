import numpy as np
import math

class NetworkLayer:
	def __init__(self, num_neurons, num_inputs, activation_fn=3, weights=None):
		if weights is None:
			self.weights = np.random.rand(num_neurons,num_inputs)
		else:
			self.weights = weights
		self.biases = np.random.rand(num_neurons)
		self.output = np.zeros(num_neurons)
		self.deltas = np.zeros(num_neurons)
		self.inputs = np.zeros(num_inputs)
		self.weights_delta = np.zeros((num_neurons,num_inputs))
		self.biases_delta = np.zeros(num_neurons)
		if activation_fn == 1:
			self.activation_fn = self.relu
		elif activation_fn == 2:
			self.activation_fn = self.stepwise_sigmoid
		elif activation_fn == 3:
			self.activation_fn = self.sigmoid
		else:
			raise ValueError('Invalid activation_fn value!')
		
	def respond(self,inputs):
		if isinstance(inputs,list):
			inputs = np.array(inputs)
		self.inputs = inputs
		sum_input = np.dot(self.weights,inputs) + self.biases
		for i in range(0,len(self.output)):
			self.output[i] = self.activation_fn(sum_input[i])
		return self.output
	
	def calc_delta(self,error,eta):
		if isinstance(error,list):
			error = np.array(error)
		deltas = error*self.output*(1-self.output)
		self.weights_delta += eta*(np.reshape(deltas,(-1,1))*self.inputs)
		self.biases_delta += eta*deltas
		return deltas
		
	def update_weights(self):
		self.weights += self.weights_delta
		self.weights_delta = np.zeros(self.weights_delta.shape)
		self.biases += self.biases_delta
		self.biases_delta = np.zeros(self.biases_delta.shape)
		
	def stepwise_sigmoid(self,x):
		y = x
		if y < 0:
			y = 0
		elif y > 1:
			y = 1
		return y

	def relu(self,x):
		y = x
		if y < 0:
			y = 0
		return y

	def sigmoid(self,x):
		return 1/(1+math.exp(-x))

