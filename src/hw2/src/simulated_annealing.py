#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:04:58 2019

@author: Zeke
"""
import numpy as np
import copy
from traveling_salesman_base import TravelingSalesmanBase

class SimulatedAnnealing(TravelingSalesmanBase):
	def __init__(self,x,y):
		# Initialize temperature T and calculate the cost matrix
		TravelingSalesmanBase.__init__(self,x,y)
		self.T = 100
		self.T_delta = self.T/100000
		
		# Initialize a random state
		self.state = np.random.permutation(len(x))
		self.cost = self.calc_cost(self.state)
		
		# Save the current state as the best state
		self.best_state = copy.deepcopy(self.state)
		self.best_cost = copy.deepcopy(self.cost)
		
		self.num_solutions_generated = 1
	
	def iterate(self):
		# Randomly generate successor state
		new_state = self.generate_successor_state()
		new_cost = self.calc_cost(new_state)
		if new_cost < self.cost:
			self.state = new_state
			self.cost = new_cost
			self.update_best_state()
		else:
			random = np.random.rand(1)
			delta_d = new_cost - self.cost
			
			if np.exp(-delta_d/self.T) > random:
				self.state = new_state
				self.cost = new_cost
				
		self.temperature_schedule()
		self.num_solutions_generated += 1
		
	def temperature_schedule(self):
		self.T -= self.T_delta
		if self.T < 0.0001:
			self.T = 0.0001
		
	def update_best_state(self):
		if self.cost < self.best_cost:
			self.best_cost = copy.deepcopy(self.cost)
			self.best_state = copy.deepcopy(self.state)
#			print(self.best_cost, self.num_solutions_generated)
		
	def generate_successor_state(self):
		# Mutation method - randomly pick two entries in the solution and swap them
		cities_to_swap = np.random.randint(len(self.x),size=2)
		new_state = copy.deepcopy(self.state)
		new_state[cities_to_swap[0]], new_state[cities_to_swap[1]] = new_state[cities_to_swap[1]], new_state[cities_to_swap[0]]
		return new_state