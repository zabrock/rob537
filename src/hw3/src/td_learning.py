#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:44:42 2019

@author: Zeke
"""
import numpy as np
import random
import matplotlib.pyplot as plt
from gridworld import Gridworld, Position

class TDLearning(object):
	def __init__(self,board,alpha=0.1,epsilon=0.05,gamma=0.9, 
			  move_symbol_map = {(-1,0):'L', (1,0):'R', (0,1):'U', (0,-1):'D', (0,0):'S'}):
		self.board = board
		self.alpha = alpha # Learning rate
		self.epsilon = epsilon # Probability of random exploration
		self.gamma = gamma # Time value of future rewards
		self.move_map = move_symbol_map # Key-value mapping of possible moves and their string representations
		self.allowed_moves = list(self.move_map.keys())
		
		self.V = np.random.rand(*self.board.dimensions())
		
	def run_episode(self,agent_loc,goal_loc,episode_length=20):
		reward = 0
		count = 0
		pos = agent_loc.clone()
		for step in range(episode_length):
			# With chance epsilon, pick a random move
			if np.random.random() < self.epsilon:
				chosen_move = random.choice(self.allowed_moves)
			# Otherwise pick the move with the highest value
			else:
				action_values = [self.V[self.board.move(pos, move).coordinates()] for move in self.allowed_moves]
				chosen_move = self.allowed_moves[np.argmax(action_values)]
				
			# Estimate reward as the value of the next state
			new_pos_val = self.gamma * self.V[self.board.move(pos, chosen_move).coordinates()]
			# Adjust the value of the current state
			value_increment = self.alpha*(new_pos_val - self.V[pos.coordinates()])
			if pos == goal_loc:
				value_increment += self.alpha*20 # Reward 20 for reaching the goal
				reward += 20*self.gamma**count # Add to discounted sum of rewards
			else:
				value_increment -= self.alpha # Reward -1 everywhere else
				reward -= self.gamma**count # Add to discounted sum of rewards
			self.V[pos.coordinates()] += value_increment
			count += 1
			
			# Make the selected move
			pos = self.board.move(pos, chosen_move)
			
		return reward
	
if __name__ == "__main__":
	# Initialize the grid world with appropriate goal location, size, and obstacles
	gw = Gridworld(10,5,[(7,0),(7,1),(7,2)],(9,1))
	# Initialize the QLearning object
	g = TDLearning(gw)
	# Initialize a reward vector to see how reward evolves per episode
	reward = np.zeros(1000)
	# Visualize the policy before training occurs
	gw.visualize_world(Position(9,1),g.V)
	for i in range(0,len(reward)):
		if i == 10 or i == 100:
			# Visualize policy at 10 training steps and 100 training steps
			gw.visualize_world(Position(9,1),g.V)
		reward[i] = g.run_episode(gw.random_position(),Position(9,1))
	# Visualize policy  after full set of training steps
	gw.visualize_world(Position(9,1),g.V)
	# Plot reward over time
	plt.scatter(np.arange(len(reward)),reward)
	plt.xlabel('Episode')
	plt.ylabel('Discounted sum of rewards')
	plt.show()