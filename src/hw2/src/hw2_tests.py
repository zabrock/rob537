#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:04:58 2019

@author: Zeke
"""
import pandas as pd

from visualize_ts import VisualizeTS
from simulated_annealing import SimulatedAnnealing

def load_dataset(csv_file):
	var_names = ['x','y']
	df = pd.read_csv(csv_file,header=None,names=var_names)
	return df['x'].values, df['y'].values
	
if __name__ == "__main__":
	x, y = load_dataset('hw2_data/15cities.csv')
	anneal = SimulatedAnnealing(x, y)
	viz = VisualizeTS(x, y)

	anneal.calc_cost_matrix()
	for i in range(0,1000000):
		anneal.iterate()
		if i in [0,99,999,9999,99999,999999]:
			print(anneal.best_cost)
			viz.plot_solution(anneal.best_state)