#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:16:13 2019

@author: Zeke
"""
from networklayer import NetworkLayer
import numpy as np
import pandas as pd

class MultiLayerNetwork:
	def __init__(self,n_inputs,n_hidden,n_outputs,eta,hidden_weights=None,output_weights=None):
		self.layers = [NetworkLayer(n_hidden, n_inputs,weights=hidden_weights), NetworkLayer(n_outputs, n_hidden,weights=output_weights)]
		self.output = None
		self.eta = eta
		self.data = {}
		
	def respond(self,inputs):
		temp_inputs = inputs
		for layer in self.layers:
			temp_inputs = layer.respond(temp_inputs)
			
		self.output = temp_inputs
		
		return self.output
	
	def calc_layer_deltas(self,output_error):
		layer_error = output_error
		for layer in reversed(self.layers):
			layer_deltas = layer.calc_delta(layer_error,self.eta)
			layer_error = np.matmul(layer_deltas,layer.weights)
			
	def update_layer_weights(self):
		for layer in self.layers:
			layer.update_weights()
			
	def read_dataset(self,csv_file):
		var_names = ['x1','x2','y1','y2']
		df = pd.read_csv(csv_file,header=None,names=var_names)
		self.data['x1'], self.data['x2'], self.data['y1'], self.data['y2'] = \
			df['x1'].values, df['x2'].values, df['y1'].values, df['y2'].values
	
	def train(self,data_idx):
		inputs = np.array([self.data['x1'][data_idx],self.data['x2'][data_idx]])
		output = self.respond(inputs)
		des = np.array([self.data['y1'][data_idx],self.data['y2'][data_idx]])
		error = des - output
		self.calc_layer_deltas(error)
		return error
	
	def train_epoch(self,update_rate=1):
		if update_rate < 1:
			update_rate = 1
			raise Warning('Update rate was entered as less than every iteration (<1); changed to update every iteration')
			
		overall_error = 0
		for i in range(0,len(self.data['x1'])):
			error = self.train(i)
			if not i % update_rate:
				self.update_layer_weights()
			overall_error += 0.5*np.sum(error**2)
		
		return overall_error
	
	def test_network(self,test_csv):
		self.read_dataset(test_csv)
		error_overall = 0
		success = np.zeros(len(self.data['x1']))
		for data_idx in range(0,len(self.data['x1'])):
			inputs = np.array([self.data['x1'][data_idx],self.data['x2'][data_idx]])
			output = self.respond(inputs)
			des = np.array([self.data['y1'][data_idx],self.data['y2'][data_idx]])
			error_overall += 0.5*np.sum((des-output)**2)
			success[data_idx] = np.argmax(output) == np.argmax(des)
				
		percent_success = np.sum(success)/success.size
		return percent_success, error_overall
			
#def test():
#	net = MultiLayerNetwork(2,5,2,0.1)
#	iterations = 10000
#	error_overall = np.zeros(iterations)
#	for i in range(0,iterations):
#		inputs = np.array([3,4])
#		output = net.respond(inputs)
#		des = np.array([1,0])
#		error = des - output
#		net.calc_layer_deltas(error)
#		net.update_layer_weights()
#		error_overall[i] = 0.5*np.sum(error**2)
#		
#	plt.plot(np.arange(iterations),error_overall)
#	plt.show()
#	print(error)
#	
#def test_training():
#	net = MultiLayerNetwork(2,5,2,0.1)
#	net.read_dataset('data/train1.csv')
#	iterations = 10000
#	error_overall = np.zeros(iterations)
#	for i in range(0,iterations):
#		error = net.train(0)
#		error_overall[i] = 0.5*np.sum(error**2)
#		net.calc_layer_deltas(error)
#		net.update_layer_weights()
#		
#	plt.plot(np.arange(iterations),error_overall)
#	plt.show()
#	
#def test_epoch_training():
#	net = MultiLayerNetwork(2,5,2,0.1)
#	net.read_dataset('data/train1.csv')
#	iterations = 500
#	error_overall = np.zeros(iterations)
#	for i in range(0,iterations):
#		error_overall[i] = net.train_epoch()
#		
#	plt.plot(np.arange(iterations),error_overall)
#	plt.show()
	

	
	