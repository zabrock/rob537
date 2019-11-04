#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:04:58 2019

@author: Zeke
"""
import pandas as pd

from visualize_ts import VisualizeTS
from simulated_annealing import SimulatedAnnealing
from evolutionary_algorithm import EvolutionaryAlgorithm
from stochastic_beam_search import StochasticBeamSearch
import time
import copy
import matplotlib.pyplot as plt
import math

def load_dataset(csv_file):
	var_names = ['x','y']
	df = pd.read_csv(csv_file,header=None,names=var_names)
	return df['x'].values, df['y'].values
			
def tsp_solver(dataset_csv, algorithm, max_iter, num_solutions):
	# Load dataset and initialize algorithm; this also initializes the random
	# solutions and computes the cost matrix
	x, y = load_dataset(dataset_csv)
	viz = VisualizeTS(x, y)
	if algorithm == SimulatedAnnealing:
		alg = algorithm(x,y)
		# Increase the maximum iterations for SimulatedAnnealing by factor of 
		# num_solutions so that it sees as many new solutions as do the other
		# two methods before deciding it's finished
		max_iter = max_iter * num_solutions
	else:
		alg = algorithm(x, y, num_solutions)
		
	# Limit StochasticBeamSearch to 100 iterations since it seems from testing that anything
	# more than that is just unnecessary computation time
	if algorithm == StochasticBeamSearch:
		max_iter = 100
		
	iter_before_cost_change = 0
	total_iter = 0
	min_cost = copy.deepcopy(alg.best_cost)
	
	start_time = time.time()
	while iter_before_cost_change < max_iter and time.time() - start_time < 60:
		alg.iterate()
		if algorithm == SimulatedAnnealing and alg.cost != min_cost:
			min_cost = copy.deepcopy(alg.cost)
			iter_before_cost_change = 0
			
		elif algorithm != SimulatedAnnealing and alg.best_cost < min_cost:
			
			min_cost = copy.deepcopy(alg.best_cost)
			iter_before_cost_change = 0
		else:
			iter_before_cost_change += 1
			
		total_iter += 1
		
	total_time = time.time() - start_time
	if algorithm == SimulatedAnnealing:
		min_cost = alg.best_cost
		
	viz.plot_solution(alg.best_state,title=str(algorithm)+" "+str(dataset_csv))
	return min_cost, total_time, alg.num_solutions_generated

def test_suite(num_runs=10, max_iter=10000, pool_count=5):
	algorithms = [SimulatedAnnealing, EvolutionaryAlgorithm, StochasticBeamSearch]
	test_sets = ['hw2_data_v2/15cities.csv','hw2_data_v2/25cities.csv','hw2_data_v2/25cities_a.csv','hw2_data_v2/100cities.csv']
	
	all_costs = {}
	all_times = {}
	all_num_solns = {}
	
	for test_set in test_sets:
		print(test_set)
		set_costs = []
		set_times = []
		set_num_solns = []
		for algorithm in algorithms:
			print(algorithm)
			costs = []
			times = []
			num_solns = []
			for i in range(0,num_runs):
				cost, solve_time, num_soln = tsp_solver(test_set, algorithm, max_iter, pool_count)
				costs.append(cost)
				times.append(solve_time)
				num_solns.append(num_soln)
				print(i)
			set_costs.append(costs)
			set_times.append(times)
			set_num_solns.append(num_solns)
			
		all_costs[test_set] = set_costs
		all_times[test_set] = set_times
		all_num_solns[test_set] = set_num_solns
		plt.boxplot(set_costs)
		plt.ylabel('Cost')
		plt.title(test_set)
		plt.xticks([1, 2, 3], ['Annealing','Evolutionary','Stochastic Beam'])
		plt.savefig('cost_'+test_set[12:-4])
		plt.show()
		
		time.sleep(1)
		plt.boxplot(set_times)
		plt.ylabel('Solve Time (s)')
		plt.title(test_set)
		plt.xticks([1, 2, 3], ['Annealing','Evolutionary','Stochastic Beam'])
		plt.savefig('time_'+test_set[12:-4])
		plt.show()
		
		time.sleep(1)
		plt.boxplot(set_num_solns)
		plt.ylabel('Number of solutions explored')
		plt.title(test_set)
		plt.xticks([1, 2, 3], ['Annealing','Evolutionary','Stochastic Beam'])
		plt.savefig('num_solns_'+test_set[12:-4])
		plt.show()
		
	fig, ax = plt.subplots()
	anneal_solns = [all_num_solns[test_set][0] for test_set in test_sets]
	evolution_solns = [all_num_solns[test_set][1] for test_set in test_sets]
	stochastic_solns = [all_num_solns[test_set][2] for test_set in test_sets]
	bpl = plt.boxplot(anneal_solns, positions=[1,6,11,16], sym='', widths=0.8)
	bpc = plt.boxplot(evolution_solns, positions=[2,7,12,17], sym='', widths=0.8)
	bpr = plt.boxplot(stochastic_solns, positions=[3,8,13,18], sym='', widths=0.8)
	plt.xlim(0,19)
	set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
	set_box_color(bpc, '#2CA25F')
	set_box_color(bpr, '#2C7BB6')
		
	# draw temporary red and blue lines and use them to create a legend
	plt.plot([], c='#D7191C', label='Annealing')
	plt.plot([], c='#2CA25F', label='Evolutionary')
	plt.plot([], c='#2C7BB6', label='Stochastic Beam')
	plt.legend()
		
	ax.set_xticklabels(['15','25', '25a', '100'])
	ax.set_xticks([2, 7, 12, 17])
	plt.ylabel('Number of solutions')
	plt.xlabel('Number of cities')
	plt.title('Solutions dexplored per problem')
	plt.tight_layout()
	plt.savefig('solution_comparison.png')
	plt.show()
	
	fig, ax = plt.subplots()
	num_cities = [15, 25, 25, 100]
	anneal_pct_solns = [[num_soln/math.factorial(cities) for num_soln in soln] for cities, soln in zip(num_cities, anneal_solns)]
	evolution_pct_solns = [[num_soln/math.factorial(cities) for num_soln in soln] for cities, soln in zip(num_cities, evolution_solns)]
	stochastic_pct_solns = [[num_soln/math.factorial(cities) for num_soln in soln] for cities, soln in zip(num_cities, stochastic_solns)]
	bpl = plt.boxplot(anneal_pct_solns, positions=[1,6,11,16], sym='', widths=0.8)
	bpc = plt.boxplot(evolution_pct_solns, positions=[2,7,12,17], sym='', widths=0.8)
	bpr = plt.boxplot(stochastic_pct_solns, positions=[3,8,13,18], sym='', widths=0.8)
	plt.xlim(0,19)
	set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
	set_box_color(bpc, '#2CA25F')
	set_box_color(bpr, '#2C7BB6')
		
	# draw temporary red and blue lines and use them to create a legend
	plt.plot([], c='#D7191C', label='Annealing')
	plt.plot([], c='#2CA25F', label='Evolutionary')
	plt.plot([], c='#2C7BB6', label='Stochastic Beam')
	plt.legend()
		
	ax.set_xticklabels(['15','25', '25a', '100'])
	ax.set_xticks([2, 7, 12, 17])
	plt.ylabel('Percent of total solutions explored')
	plt.xlabel('Number of cities')
	plt.title('Percent of solutions explored per problem')
	plt.tight_layout()
	plt.savefig('solution_percentage_comparison.png')
	plt.show()
	
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)
	
def plot_25_cities():
	sets = ['hw2_data_v2/25cities.csv','hw2_data_v2/25cities_a.csv']
	for dataset in sets:
		x, y = load_dataset(dataset)
		viz = VisualizeTS(x, y)
		viz.plot_cities()

	
if __name__ == "__main__":
	test_suite()
	plot_25_cities()
	print(format(math.factorial(15),'E'))
	print(format(math.factorial(25),'E'))
	print(format(math.factorial(100),'E'))
	