import numpy as np
import math

class Neuron:
	def __init__(self, input_neurons=None, weights=None,activation_fn=1):
		self.output = 0.0
		self.weights = weights
		self.input_neurons = input_neurons
		if activation_fn == 1:
			self.activation_fn = self.relu
		elif activation_fn == 2:
			self.activation_fn = self.stepwise_sigmoid
		elif activation_fn == 3:
			self.activation_fn = self.sigmoid
		else:
			raise ValueError('Invalid activation_fn value!')
		
	def respond(self):
		current_inputs = [neuron.output for neuron in self.input_neurons]
		print self.weights, current_inputs
		sum_input = np.sum(np.multiply(self.weights,current_inputs))
		self.output = self.activation_fn(sum_input)
		print self.output

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

