from neuron import Neuron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Network:
	def __init__(self,n_inputs,n_hidden,n_outputs):
		'''
		Initialization function.
		Args:
			
		'''
		self.input_layer = [Neuron() for i in range(0,n_inputs)]
		self.hidden_layer = [Neuron(input_neurons=self.input_layer,weights=np.random.rand(n_inputs)) for i in range(0,n_hidden)]
		self.output_layer = [Neuron(input_neurons=self.hidden_layer,weights=np.random.rand(n_hidden)) for i in range(0,n_outputs)]
		self.data = []

	def respond(self,input_values):
		for in_value, input_neuron in zip(input_values, self.input_layer):
			input_neuron.output = in_value
		for hidden_neuron in self.hidden_layer:
			hidden_neuron.respond()
		for output_neuron in self.output_layer:
			output_neuron.respond()

	def print_output(self):
		for idx in range(0,len(self.output_layer)):
			print('Output {}: {}'.format(idx,self.output_layer[idx].output))
			
	def read_dataset(self,csv_file):
		var_names = ['x1','x2','y1','y2']
		df = pd.read_csv(csv_file,header=None,names=var_names)
		return df['x1'].values, df['x2'].values, df['y1'].values, df['y2'].values
	
	def train(self,training_csv):
		self.data.x1, self.data.x2, self.data.y1, self.data.y2 = self.read_dataset(training_csv)
		

def test():
	net = Network(1,2,2)
	net.respond([3])
	net.print_output()
	
def main():
	net = Network(2,3,2)
	x1, x2, y1, y2 = net.read_dataset('data/train1.csv')
	plt.plot(x1[y1 == 0], x2[y1 == 0], 'ro')
	plt.plot(x1[y1 == 1], x2[y1 == 1], 'go')
	plt.show()
#	for x1val, x2val in zip(x1,x2):
#		net.respond([x1val,x2val])
#		net.print_output()
	

if __name__=='__main__':
	main()
	
