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
	popl = StochasticBeamSearch(x, y, 2)
	popl.iterate()
	
	
if __name__ == "__main__":
	test_population()
	