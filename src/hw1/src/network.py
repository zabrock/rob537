from neuron import Neuron
import numpy as np

class Network:
	def __init__(self,n_inputs,n_hidden,n_outputs):
		self.input_layer = [Neuron() for i in range(0,n_inputs)]
		self.hidden_layer = [Neuron(input_neurons=self.input_layer,weights=np.random.rand(n_inputs)) for i in range(0,n_hidden)]
		self.output_layer = [Neuron(input_neurons=self.hidden_layer,weights=np.random.rand(n_outputs)) for i in range(0,n_outputs)]

	def respond(self,input_values):
		for in_value, input_neuron in zip(input_values, self.input_layer):
			input_neuron.output = in_value
		for hidden_neuron in self.hidden_layer:
			hidden_neuron.respond()
		for output_neuron in self.output_layer:
			output_neuron.respond()

	def print_output(self):
		for idx in range(0,len(self.output_layer)):
			print 'Output {}: {}'.format(idx,self.output_layer[idx].output)

def test():
	net = Network(1,2,2)
	net.respond([3])
	net.print_output()

if __name__=='__main__':
	test()
