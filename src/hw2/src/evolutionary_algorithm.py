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
	def __init__(self,x,y,num_states,crossover_probability=0.5):
		# Initialize TS Base class which calculates the cost matrix
		TravelingSalesmanBase.__init__(self,x,y)
		
		# Initialize population of random states
		self.states = [np.random.permutation(len(x)) for i in range(0,num_states)]
		self.state_costs = np.array([self.calc_cost(state) for state in self.states])
		
		# Save the current state as the best state
		self.best_state = copy.deepcopy(self.states[np.argmin(self.state_costs)])
		self.best_cost = copy.deepcopy(np.amin(self.state_costs))
		
		# Save crossover probability
		self.crossover_probability = crossover_probability
		# Save number of states to keep through each iteration
		self.num_states = num_states
	
	def iterate(self):
		# Randomly generate successor states
		successors = self.generate_successor_states()
		# Perturb successor states
		for successor in successors:
			successor = self.mutate_state(successor)
		# Update best solution so far
		successor_costs = np.array([self.calc_cost(state) for state in successors])
		self.states.extend(successors)
		self.state_costs = np.append(self.state_costs, successor_costs)
		self.update_best_state()
		# Select next generation from successor states and parent states
		self.select_next_generation()
		
		
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
		return successor_states
		
	def mutate_state(self,state):
		# Mutation method - randomly pick two entries in the solution and swap them
		cities_to_swap = np.random.randint(len(state),size=2)
		state[cities_to_swap[0]], state[cities_to_swap[1]] = state[cities_to_swap[1]], state[cities_to_swap[0]]
		return state
	
	def random_state_index(self,selection_probability):
		p = np.random.rand(1)
		return np.argmax(selection_probability > p)
	
	def select_random_state(self,selection_probability):
		# Probabilistically pick a state from solution set by using selection probability
		return copy.deepcopy(self.states[self.random_state_index(selection_probability)])
	
	def crossover_random_states(self,selection_probability):
		# Crossover method - randomly pick two states from candidate set and 
		# combine them
		solutions_to_cross = [self.select_random_state(selection_probability) for idx in range(0,2)]
		# Randomly select range of indices which should be inserted from second solution into first\
		cross_idxs = np.random.randint(len(solutions_to_cross[0]),size=2)
		# Randomly select insertion point into first solution
		insertion_idx = np.random.randint(len(solutions_to_cross[0]))
		# Split first solution at insertion point
		new_soln = np.split(solutions_to_cross[0],[insertion_idx])
		new_insert = solutions_to_cross[1][np.amin(cross_idxs):np.amax(cross_idxs)]
		# Remove duplicates in new_insert from split first solution
		new_soln = [partial_soln[np.isin(partial_soln,new_insert,invert=True)] for partial_soln in new_soln]
		# Insert cut portion from second solution into new solution
		new_soln.insert(1,new_insert)
		return np.concatenate(new_soln)