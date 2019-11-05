#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:44:42 2019

@author: Zeke
"""
import numpy as np
#import random

from gridworld import Gridworld, Position
import matplotlib.pyplot as plt
import time

class QLearning(object):
	def __init__(self,board,alpha=0.1,epsilon=0.05,gamma=0.9, 
			  move_symbol_map = {(-1,0):'L', (1,0):'R', (0,1):'U', (0,-1):'D', (0,0):'S'}):
		self.board = board
		self.alpha = alpha # Learning rate
		self.epsilon = epsilon # Probability of random exploration
		self.gamma = gamma # Time value of future rewards
		self.move_map = move_symbol_map # Key-value mapping of possible moves and their string representations
		self.allowed_moves = list(self.move_map.keys())
		
		self.Q = np.random.rand(*self.board.dimensions(),len(self.allowed_moves))
		
	def run_episode(self,agent_loc,goal_loc,episode_length=20,move_goal=False):
		reward = 0
		count = 0
		pos = agent_loc.clone()
		for step in range(episode_length):
			# Let the goal move this time step if move_goal is true
			if move_goal:
				goal_loc = self.board.move_goal()
			# With chance epsilon, pick a random move
			coords = pos.coordinates()
			if np.random.random() < self.epsilon:
				chosen_move_idx = np.random.randint(len(self.allowed_moves))
			# Otherwise pick the move with the highest value
			else:
				action_values = [self.Q[coords[0],coords[1],move] for move in range(0,len(self.allowed_moves))]
				chosen_move_idx = np.argmax(action_values)
			chosen_move = self.allowed_moves[chosen_move_idx]
				
			# Estimate reward as the value of the best possible next state-action pair
			new_pos = self.board.move(pos, chosen_move)
			new_coords = new_pos.coordinates()
			next_action_vals = [self.Q[new_coords[0],new_coords[1],move] for move in range(0,len(self.allowed_moves))]
			new_pos_val = self.gamma * np.amax(next_action_vals)
			# Adjust the value of the current state
			value_increment = self.alpha*(new_pos_val - self.Q[coords[0],coords[1],chosen_move_idx])
			if new_pos == goal_loc:
				value_increment += self.alpha*20 # Reward 20 for reaching the goal
				reward += 20*self.gamma**count # Add to discounted sum of rewards
			else:
				value_increment -= self.alpha # Reward -1 everywhere else
				reward -= self.gamma**count # Add to discounted sum of rewards
			self.Q[coords[0],coords[1],chosen_move_idx] += value_increment
			count += 1
			
			# Make the selected move
			pos = self.board.move(pos, chosen_move)
			
#		self.board.visualize_world(pos,goal_loc)
		return reward
	
if __name__ == "__main__":
	# Initialize the grid world with appropriate goal location, size, and obstacles
	gw = Gridworld(10,5,[(7,0),(7,1),(7,2)],(9,1))
	# Initialize the QLearning object
	g = QLearning(gw)
	# Initialize a reward vector to see how reward evolves per episode
	reward = np.zeros(1000)
	# Visualize the policy before training occurs
	gw.visualize_world(V=g.Q)
	for i in range(0,len(reward)):
		if i == 10 or i == 100:
			# Visualize policy at 10 training steps and 100 training steps
			gw.visualize_world(V=g.Q)
		reward[i] = g.run_episode(gw.random_position(),Position(9,1))
	# Visualize policy  after full set of training steps
	gw.visualize_world(V=g.Q)
	# Plot reward over time
	plt.scatter(np.arange(len(reward)),reward)
	plt.xlabel('Episode')
	plt.ylabel('Discounted sum of rewards')
	plt.show()
	
	# Now repeat the above using a second QLearning instance with a moving target
	g_move = QLearning(gw)
	# Initialize a reward vector to see how reward evolves per episode
	reward_move = np.zeros(10000)
	# Visualize the policy before training occurs
	gw.visualize_world(V=g_move.Q)
	for i in range(0,len(reward_move)):
		if i == 10 or i == 100 or i == 1000:
			# Visualize policy at 10 training steps and 100 training steps
			gw.visualize_world(V=g_move.Q)
		reward_move[i] = g_move.run_episode(gw.random_position(),Position(9,1),move_goal=True)
	# Visualize policy  after full set of training steps
	gw.visualize_world(V=g_move.Q)
	# Plot reward over time
	plt.scatter(np.arange(len(reward_move)),reward_move)
	plt.xlabel('Episode')
	plt.ylabel('Discounted sum of rewards')
	plt.show()
	
	fig, ax = plt.subplots()
	plt.boxplot([reward,reward_move[0:len(reward)]])
	ax.set_xticklabels(['Static target','Moving target'])
	plt.ylabel('Discounted sum of rewards')