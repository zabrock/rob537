import numpy as np
from multilayernetwork import MultiLayerNetwork
import matplotlib.pyplot as plt

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
	
def update_rate_impact():
	update_rate_options = [1,2,5,10,20,50]

	units_error1 = []
	units_accuracy1 = []
	units_error2 = []
	units_accuracy2 = []
	units_error3 = []
	units_accuracy3 = []
	training_error = []
	hidden_weights = np.random.rand(5,2)
	output_weights = np.random.rand(2,5)
	for update_rate in update_rate_options:
		print('update_rate: ',update_rate)
		net = MultiLayerNetwork(2,5,2,0.1,hidden_weights=hidden_weights,output_weights=output_weights)
		net.read_dataset('data/train1.csv')
		iterations = 500
		error_overall = np.zeros(iterations)
		for i in range(0,iterations):
			error_overall[i] = net.train_epoch(update_rate)
			
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
	plt.legend(['Update every '+str(i)+ ' sample(s)' for i in update_rate_options],loc='upper right')
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
	plt.xlabel('Number of samples between weight updates', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(units_error1))], [str(i) for i in update_rate_options])
	plt.ylabel('Sum square error', fontweight='bold')
	
	plt.legend()
	plt.show()
	
	# Make the plot
	plt.bar(r1, units_accuracy1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Test 1')
	plt.bar(r2, units_accuracy2, color='#557f2d', width=barWidth, edgecolor='white', label='Test 2')
	plt.bar(r3, units_accuracy3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Test 3')
		 
	# Add xticks on the middle of the group bars
	plt.xlabel('Number of samples between weight updates', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(units_error1))], [str(i) for i in update_rate_options])
	plt.ylabel('Correct classification percentage', fontweight='bold')
	
	# Create legend & Show graphic
	plt.legend()
	plt.show()
	
def initial_weight_variability():
	errors = []
	for i in range(0,3):
		net = MultiLayerNetwork(2,5,2,0.1)
		net.read_dataset('data/train1.csv')
		iterations = 500
		error_overall = np.zeros(iterations)
		for i in range(0,iterations):
			error_overall[i] = net.train_epoch()
		errors.append(error_overall)
		
	for error in errors:
		plt.plot(np.arange(1,iterations+1),error)
		
	plt.xlabel('Number of epoch iterations')
	plt.ylabel('Sum square error', fontweight='bold')
	plt.xscale('log')
	plt.legend(['Network '+str(i+1) for i in range(0,3)],loc='upper right')
	plt.show()
			
	
def plot_datasets():
	net = MultiLayerNetwork(2,3,2,0.1)
	datasets = ['data/train1.csv','data/test1.csv','data/test2.csv','data/test3.csv']
	for dataset in datasets:
		net.read_dataset(dataset)
		plt.plot(net.data['x1'][net.data['y1'] == 0], net.data['x2'][net.data['y1'] == 0], 'go')
		plt.plot(net.data['x1'][net.data['y1'] == 1], net.data['x2'][net.data['y1'] == 1], 'ro')
		plt.legend(['pass','fail'])
		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.show()
	
if __name__ == '__main__':
	plot_datasets()
	hidden_units_impact()
	training_steps_impact()
	learning_rate_impact()
	update_rate_impact()
	initial_weight_variability()