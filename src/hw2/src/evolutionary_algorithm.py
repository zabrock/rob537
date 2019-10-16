#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:04:58 2019

@author: Zeke
"""
import numpy as np
import copy
from traveling_salesman_base import TravelingSalesmanBase

class EvolutionaryAlgorithm(TravelingSalesmanBase):
	def __init__(self,x,y,num_init_states):
		# Initialize temperature T and calculate the cost matrix
		TravelingSalesmanBase.__init__(self,x,y,crossover_probability=0.5)
		
		# Initialize population of random states
		self.states = [np.random.permutation(len(x)) for i in range(0,num_init_states)]
		self.costs = np.array([self.calc_cost(state) for state in self.states])
		
		# Save the current state as the best state
		self.best_state = copy.deepcopy(self.states[np.argmin(self.costs)])
		self.best_cost = copy.deepcopy(np.amin(self.costs))
		
		# Save crossover probability
		self.crossover_probability = crossover_probability
	
	def iterate(self):
		# Randomly generate successor states
		# Perturb successor states
		# Select next generation from successor states and parent states
		
	def update_best_state(self):
		best_current_cost = np.amin(self.costs)
		if best_current_cost < self.best_cost:
			self.best_cost = best_current_cost
			self.best_state = copy.deepcopy(self.states[np.argmin(self.costs)])
			
	def calculate_selection_probability(self):
		# Calculate fitness of each solution as inverse of cost
		fitness = 1/self.costs
		# Use roulette wheel (fitness proportional) probability
		return np.cumsum(fitness/np.sum(fitness))
	
	def generate_new_state(self,selection_probability):
		# Probabilistically generate new state either by crossover or direct copy for later mutation
		p = np.random.rand(1)
		if p < self.crossover_probability:
			return self.crossover_random_states(selection_probability)
		else:
			return self.select_random_state(selection_probability)
	
	def generate_successor_states(self):
		selection_probability = self.calculate_selection_probability()
		successor_states = [self.generate_new_state(selection_probability) for state in self.states]
		
	def mutate_state(self,state):
		# Mutation method - randomly pick two entries in the solution and swap them
		cities_to_swap = np.random.randint(len(state),size=2)
		state[cities_to_swap[0]], state[cities_to_swap[1]] = state[cities_to_swap[1]], state[cities_to_swap[0]]
		return state
	
	def select_random_state(self,selection_probability):
		# Probabilistically pick a state from solution set by using selection probability
		p = np.random.rand(1)
		return self.states[np.argmax(selection_probability > p)]
	
	def crossover_random_states(self,selection_probability):
		# Crossover method - randomly pick two states from candidate set and 
		# combine them
		solutions_to_cross = [self.select_random_state(selection_probability) for idx in range(0,2)]