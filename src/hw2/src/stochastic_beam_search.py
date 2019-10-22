#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:04:58 2019

@author: Zeke
"""
import numpy as np
import copy
from traveling_salesman_base import TravelingSalesmanBase
from itertools import combinations

class StochasticBeamSearch(TravelingSalesmanBase):
	def __init__(self,x,y,num_states):
		# Initialize TS Base class which calculates the cost matrix
		TravelingSalesmanBase.__init__(self,x,y)
		
		# Initialize population of random states
		self.states = [np.random.permutation(len(x)) for i in range(0,num_states)]
		self.state_costs = np.array([self.calc_cost(state) for state in self.states])
		
		# Save the current state as the best state
		self.best_state = copy.deepcopy(self.states[np.argmin(self.state_costs)])
		self.best_cost = copy.deepcopy(np.amin(self.state_costs))
		
		# Save number of states to keep through each iteration
		self.num_states = num_states
		
		# Precompute all possible permutation indices for a given state
		self.mut_idx = [list(x) for x in combinations(range(0,len(x)),2)]
		
	def iterate(self):
		# Generate all successor states for all current states
		self.generate_successor_states()
		# Update best solution so far
		self.update_best_state()
		# Select next generation from successor states and parent states
		self.select_next_generation()
		
	def generate_successor_states(self):
		# Loop through all current states and generate successors for each
		successors = []
		for state in self.states:
			# Find all possible single-swap mutations for the state
			successors.extend(self.all_state_mutations(state))
		# Add successors and their costs to the solution pool
		successor_costs = np.array([self.calc_cost(state) for state in successors])
		self.states.extend(successors)
		self.state_costs = np.append(self.state_costs, successor_costs)
		self.states.extend(successors)
		
	def all_state_mutations(self,state):
		# Go through all the permutation indices and swap them in copies of state
		all_new_states = []
		for cities_to_swap in self.mut_idx:
			new_state = copy.deepcopy(state)
			new_state[cities_to_swap[0]], new_state[cities_to_swap[1]] = new_state[cities_to_swap[1]], new_state[cities_to_swap[0]]
			all_new_states.append(new_state)
			
		return all_new_states
		
	def update_best_state(self):
		best_current_cost = np.amin(self.state_costs)
		if best_current_cost < self.best_cost:
			self.best_cost = best_current_cost
			self.best_state = copy.deepcopy(self.states[np.argmin(self.state_costs)])
			
	def calculate_selection_probability(self):
		# Calculate fitness of each solution as inverse of cost
		fitness = 1/self.state_costs
		# Use roulette wheel (fitness proportional) probability
		return np.cumsum(fitness/np.sum(fitness))
	
	def select_next_generation(self):
		next_gen = []
		next_gen_costs = []
		for i in range(0,self.num_states):
			selection_probability = self.calculate_selection_probability()
			# Probabilistically select state for next generation
			idx = self.random_state_index(selection_probability)
			next_gen.append(self.states.pop(idx))
			next_gen_costs.append(copy.deepcopy(self.state_costs[idx]))
			self.state_costs = np.delete(self.state_costs,idx)
		self.states = next_gen
		self.state_costs = np.array(next_gen_costs)
		
	def random_state_index(self,selection_probability):
		p = np.random.rand(1)
		return np.argmax(selection_probability > p)