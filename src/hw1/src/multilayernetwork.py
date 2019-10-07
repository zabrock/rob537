#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:16:13 2019

@author: Zeke
"""
from networklayer import NetworkLayer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class MultiLayerNetwork:
	def __init__(self,n_inputs,n_hidden,n_outputs,eta):
		self.layers = [NetworkLayer(n_hidden, n_inputs), NetworkLayer(n_outputs, n_hidden)]
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
			layer_deltas = layer.calc_delta(layer_error)
			layer_error = np.matmul(layer_deltas,layer.weights)
			
	def backpropagate(self):
		for layer in self.layers:
			layer.update_weights(self.eta)
			
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
	
	def train_epoch(self):
		overall_error = 0
		for i in range(0,len(self.data['x1'])):
			error = self.train(i)
			self.calc_layer_deltas(error)
#			if not i % 50:
#				print(error)
			self.backpropagate()
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
			
def test():
	net = MultiLayerNetwork(2,5,2,0.1)
	iterations = 10000
	error_overall = np.zeros(iterations)
	for i in range(0,iterations):
		inputs = np.array([3,4])
		output = net.respond(inputs)
		des = np.array([1,0])
		error = des - output
		net.calc_layer_deltas(error)
		net.backpropagate()
		error_overall[i] = 0.5*np.sum(error**2)
		
	plt.plot(np.arange(iterations),error_overall)
	plt.show()
	print(error)
	
def test_training():
	net = MultiLayerNetwork(2,3,2,0.1)
	net.read_dataset('data/train1.csv')
	iterations = 10000
	error_overall = np.zeros(iterations)
	for i in range(0,iterations):
		error = net.train(0)
		error_overall[i] = 0.5*np.sum(error**2)
		net.calc_layer_deltas(error)
		net.backpropagate()
		
	plt.plot(np.arange(iterations),error_overall)
	plt.show()
	
def test_epoch_training():
	net = MultiLayerNetwork(2,5,2,0.1)
	net.read_dataset('data/train1.csv')
	iterations = 500
	error_overall = np.zeros(iterations)
	for i in range(0,iterations):
		error_overall[i] = net.train_epoch()
		
	plt.plot(np.arange(iterations),error_overall)
	plt.show()
	
def hidden_units_impact():
	units_options = np.arange(2,32,step=3)
	units_error1 = []
	units_accuracy1 = []
	units_error2 = []
	units_accuracy2 = []
	units_error3 = []
	units_accuracy3 = []
	training_error = []
	for num_units in units_options:
		print('Num_units: ',num_units)
		net = MultiLayerNetwork(2,num_units,2,0.1)
		net.read_dataset('data/train1.csv')
		iterations = 500
		error_overall = np.zeros(iterations)
		for i in range(0,iterations):
			error_overall[i] = net.train_epoch()
			
		training_error.append(error_overall)
		percent_success, error_overall = net.test_network('data/test1.csv')
		units_error1.append(error_overall)
		units_accuracy1.append(percent_success)
		percent_success, error_overall = net.test_network('data/test2.csv')
		units_error2.append(error_overall)
		units_accuracy2.append(percent_success)
		percent_success, error_overall = net.test_network('data/test3.csv')
		units_error3.append(error_overall)
		units_accuracy3.append(percent_success)
		
		
	for error in training_error:
		plt.plot(np.arange(1,iterations+1),error)
		
	plt.xlabel('Epoch training iterations', fontweight='bold')
	plt.ylabel('Sum square error', fontweight='bold')
	plt.xscale('log')
	plt.legend(units_options.astype(str),loc='upper right')
	plt.show()
	
	# set width of bar
	barWidth = 0.25
	
	# Set position of bar on X axis
	r1 = np.arange(len(units_error1))
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]
	
	# Make the plot
	plt.bar(r1, units_error1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Test 1')
	plt.bar(r2, units_error2, color='#557f2d', width=barWidth, edgecolor='white', label='Test 2')
	plt.bar(r3, units_error3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Test 3')
		 
	# Add xticks on the middle of the group bars
	plt.xlabel('Number of hidden units', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(units_error1))], units_options.astype(str))
	plt.ylabel('Sum square error', fontweight='bold')
	
	plt.legend()
	plt.show()
	
	# Make the plot
	plt.bar(r1, units_accuracy1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Test 1')
	plt.bar(r2, units_accuracy2, color='#557f2d', width=barWidth, edgecolor='white', label='Test 2')
	plt.bar(r3, units_accuracy3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Test 3')
		 
	# Add xticks on the middle of the group bars
	plt.xlabel('Number of hidden units', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(units_error1))], units_options.astype(str))
	plt.ylabel('Correct classification percentage', fontweight='bold')
	
	# Create legend & Show graphic
	plt.legend()
	plt.show()
	
def training_steps_impact():
	num_steps_options = [1,2,5,10,25,50,100,250,500,1000,2500,5000,10000]
	units_error1 = []
	units_accuracy1 = []
	units_error2 = []
	units_accuracy2 = []
	units_error3 = []
	units_accuracy3 = []
	net = MultiLayerNetwork(2,5,2,0.1)
	net.read_dataset('data/train1.csv')
	training_error = np.zeros(num_steps_options[-1])
	num_steps_idx = 0
	for i in range(0,num_steps_options[-1]):
		training_error[i] = net.train_epoch()
		if i+1 == num_steps_options[num_steps_idx]:
			print(i+1)
			percent_success, error_overall = net.test_network('data/test1.csv')
			units_error1.append(error_overall)
			units_accuracy1.append(percent_success)
			percent_success, error_overall = net.test_network('data/test2.csv')
			units_error2.append(error_overall)
			units_accuracy2.append(percent_success)
			percent_success, error_overall = net.test_network('data/test3.csv')
			units_error3.append(error_overall)
			units_accuracy3.append(percent_success)
			num_steps_idx += 1
			net.read_dataset('data/train1.csv')
		
	plt.plot(np.arange(1,num_steps_options[-1]+1),training_error)
	plt.plot(num_steps_options,units_error1)
	plt.plot(num_steps_options,units_error2)
	plt.plot(num_steps_options,units_error3)
		
	plt.xlabel('Epoch training iterations', fontweight='bold')
	plt.ylabel('Sum square error', fontweight='bold')
	plt.xscale('log')
	plt.legend(['Training set','Test set 1','Test set 2','Test set 3'],loc='upper right')
	plt.show()
	
	# set width of bar
	barWidth = 0.25
	
	# Set position of bar on X axis
	r1 = np.arange(len(units_error1))
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]
	
	# Make the plot
	plt.bar(r1, units_error1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Test 1')
	plt.bar(r2, units_error2, color='#557f2d', width=barWidth, edgecolor='white', label='Test 2')
	plt.bar(r3, units_error3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Test 3')
		 
	# Add xticks on the middle of the group bars
	plt.xlabel('Number of training epochs', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(units_error1))], [str(i) for i in num_steps_options])
	plt.ylabel('Sum square error', fontweight='bold')
	
	plt.legend()
	plt.show()
	
	# Make the plot
	plt.bar(r1, units_accuracy1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Test 1')
	plt.bar(r2, units_accuracy2, color='#557f2d', width=barWidth, edgecolor='white', label='Test 2')
	plt.bar(r3, units_accuracy3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Test 3')
		 
	# Add xticks on the middle of the group bars
	plt.xlabel('Number of training epochs', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(units_error1))], [str(i) for i in num_steps_options])
	plt.ylabel('Correct classification percentage', fontweight='bold')
	
	# Create legend & Show graphic
	plt.legend()
	plt.show()
	
def learning_rate_impact():
	learning_rate_options = [0.001,0.01,0.1,1,10]

	units_error1 = []
	units_accuracy1 = []
	units_error2 = []
	units_accuracy2 = []
	units_error3 = []
	units_accuracy3 = []
	training_error = []
	for learning_rate in learning_rate_options:
		print('learning_rate: ',learning_rate)
		net = MultiLayerNetwork(2,5,2,learning_rate)
		net.read_dataset('data/train1.csv')
		iterations = 500
		error_overall = np.zeros(iterations)
		for i in range(0,iterations):
			error_overall[i] = net.train_epoch()
			
		training_error.append(error_overall)
		percent_success, error_overall = net.test_network('data/test1.csv')
		units_error1.append(error_overall)
		units_accuracy1.append(percent_success)
		percent_success, error_overall = net.test_network('data/test2.csv')
		units_error2.append(error_overall)
		units_accuracy2.append(percent_success)
		percent_success, error_overall = net.test_network('data/test3.csv')
		units_error3.append(error_overall)
		units_accuracy3.append(percent_success)
		
		
	for error in training_error:
		plt.plot(np.arange(1,iterations+1),error)
		
	plt.xlabel('Epoch training iterations', fontweight='bold')
	plt.ylabel('Sum square error', fontweight='bold')
	plt.xscale('log')
	plt.legend(['$\epsilon$ = '+str(i) for i in learning_rate_options],loc='upper right')
	plt.show()
	
	# set width of bar
	barWidth = 0.25
	
	# Set position of bar on X axis
	r1 = np.arange(len(units_error1))
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]
	
	# Make the plot
	plt.bar(r1, units_error1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Test 1')
	plt.bar(r2, units_error2, color='#557f2d', width=barWidth, edgecolor='white', label='Test 2')
	plt.bar(r3, units_error3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Test 3')
		 
	# Add xticks on the middle of the group bars
	plt.xlabel('Learning rate', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(units_error1))], ['$\epsilon$ = '+str(i) for i in learning_rate_options])
	plt.ylabel('Sum square error', fontweight='bold')
	
	plt.legend()
	plt.show()
	
	# Make the plot
	plt.bar(r1, units_accuracy1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Test 1')
	plt.bar(r2, units_accuracy2, color='#557f2d', width=barWidth, edgecolor='white', label='Test 2')
	plt.bar(r3, units_accuracy3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Test 3')
		 
	# Add xticks on the middle of the group bars
	plt.xlabel('Learning rate', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(units_error1))], ['$\epsilon$ = '+str(i) for i in learning_rate_options])
	plt.ylabel('Correct classification percentage', fontweight='bold')
	
	# Create legend & Show graphic
	plt.legend()
	plt.show()
	
def plot_datasets():
	net = MultiLayerNetwork(2,3,2,0.1)
	datasets = ['data/train1.csv','data/test1.csv','data/test2.csv','data/test3.csv']
	for dataset in datasets:
		net.read_dataset(dataset)
		plt.plot(net.data['x1'][net.data['y1'] == 0], net.data['x2'][net.data['y1'] == 0], 'go')
		plt.plot(net.data['x1'][net.data['y1'] == 1], net.data['x2'][net.data['y1'] == 1], 'ro')
		plt.legend(['pass','fail'])
		plt.show()
	
if __name__ == '__main__':
	plot_datasets()