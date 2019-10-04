import numpy as np
import math

class NetworkLayer:
	def __init__(self, num_neurons, num_inputs, activation_fn=3):
		self.weights = np.random.rand(num_inputs,num_neurons)
		self.biases = np.random.rand(num_neurons)
		self.output = np.zeros(num_neurons)
		self.deltas = np.zeros(num_neurons)
		self.inputs = np.zeros(num_inputs)
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
		sum_input = np.sum(np.matmul(inputs,self.weights)) + self.biases
		for i in range(0,len(self.output)):
			self.output[i] = self.activation_fn(sum_input[i])
		return self.output
	
	def calc_delta(self,error):
		if isinstance(error,list):
			error = np.array(error)
		self.deltas = np.multiply(error, np.multiply(self.output, 1-self.output))
		return self.deltas
	
	def update_weights(self,eta):
		old_weight_shape = np.shape(self.weights)
		self.weights = self.weights + eta*(np.reshape(self.inputs,(-1,1))*self.deltas)
		assert old_weight_shape == np.shape(self.weights)
		self.biases = self.biases + eta*self.deltas
		
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

