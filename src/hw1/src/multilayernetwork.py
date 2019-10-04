#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:16:13 2019

@author: Zeke
"""
from networklayer import NetworkLayer
import numpy as np
import matplotlib.pyplot as plt

class MultiLayerNetwork:
	def __init__(self,n_inputs,n_hidden,n_outputs,eta):
		self.layers = [NetworkLayer(n_hidden, n_inputs), NetworkLayer(n_outputs, n_hidden)]
		self.output = None
		self.eta = eta
		
	def respond(self,inputs):
		temp_inputs = inputs
		for layer in self.layers:
			temp_inputs = layer.respond(temp_inputs)
			
		self.output = temp_inputs
		
		return self.output
	
	def calc_layer_deltas(self,output_error):
		layer_error = output_error
		for layer in reversed(self.layers):
			layer_deltas = layer.calc_delta(layer_error)
			layer_error = np.matmul(layer_deltas,np.transpose(layer.weights))
			
	def backpropagate(self,output_error):
		self.calc_layer_deltas(output_error)
		for layer in self.layers:
			layer.update_weights(self.eta)
			
def test():
	net = MultiLayerNetwork(2,5,2,0.1)
	iterations = 10000
	error_overall = np.zeros(iterations)
	for i in range(0,iterations):
		inputs = np.array([3,4])
		output = net.respond(inputs)
		des = np.array([1,0])
		error = des - output
		net.backpropagate(error)
		error_overall[i] = 0.5*np.sum(error**2)
		
	plt.plot(np.arange(iterations),error_overall)
	plt.show()
	print(error)
	
if __name__ == '__main__':
	test()