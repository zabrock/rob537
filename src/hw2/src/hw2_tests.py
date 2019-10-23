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

def load_dataset(csv_file):
	var_names = ['x','y']
	df = pd.read_csv(csv_file,header=None,names=var_names)
	return df['x'].values, df['y'].values
	
def test_annealing():
	x, y = load_dataset('hw2_data/15cities.csv')
	anneal = SimulatedAnnealing(x, y)
	viz = VisualizeTS(x, y)

	for i in range(0,1000000):
		anneal.iterate()
		if i in [0,99,999,9999,99999,999999]:
			print(anneal.best_cost)
			viz.plot_solution(anneal.best_state)
			
def test_evolutionary():
	x, y = load_dataset('hw2_data/25cities.csv')
	ev = EvolutionaryAlgorithm(x, y, 10, crossover_probability=0.5)
	viz = VisualizeTS(x,y)
	for i in range(0,10000):
		ev.iterate()
		if i in [0,99,999,9999]:
			print(ev.best_cost)
			viz.plot_solution(ev.best_state)
			
def test_population():
	x, y = load_dataset('hw2_data/15cities.csv')
	popl = StochasticBeamSearch(x, y, 5)
	viz = VisualizeTS(x,y)
	for i in range(0,1000):
		popl.iterate()
		if i in [0, 99, 999, 9999]:
			print(popl.best_cost)
			viz.plot_solution(popl.best_state)
			
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
#			print(iter_before_cost_change)
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
		
#	print(all_costs)
#	print(all_times)
#	print(all_num_solns)
	
	
if __name__ == "__main__":
#	print(tsp_solver('hw2_data/15cities.csv',EvolutionaryAlgorithm,10000,10))
	test_suite()
	