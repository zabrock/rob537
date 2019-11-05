#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:45:58 2019

@author: Zeke
"""
import numpy as np
import matplotlib.pyplot as plt
import random

class Position(object):
	def __init__(self,x,y):
		self.coords = (x,y)
		
	def coordinates(self):
		# Return coordinates
		return self.coords
	
	def move(self,move):
		# Return a new Position object moved by (x,y) in move
		x, y = self.coords
		return Position(x + move[0], y + move[1])
	
	def clone(self):
		# Make a new copy of the Position object
		return Position(self.coords[0], self.coords[1])
	
	def __eq__(self, other):
		# Override equality test to check for matching coordinate data
		return self.coords == other.coords


class Gridworld(object):
	def __init__(self,width,height,obstacle_locs,goal_loc):
		# Set a value for how to recognize obstacles
		self.obstacle = -1
		
		# Save map dimensions, mark obstacles in the grid, and save the goal location
		self.max_x = width - 1
		self.max_y = height - 1
		self.grid = np.zeros((width,height))
		for loc in obstacle_locs:
			self.grid[loc] = self.obstacle
		self.goal_loc = Position(*goal_loc)
		
		# Save possible moves for goal location
		self.move_symbol_map = {(-1,0):'L', (1,0):'R', (0,1):'U', (0,-1):'D'}
		self.possible_moves = list(self.move_symbol_map.keys())
			
	def dimensions(self):
		# Return map dimensions
		return (self.max_x+1, self.max_y+1)
			
	def move(self,agent_loc,agent_move):
		# Start by checking whether we're starting from a valid location
		if not self.test_position_validity(agent_loc):
			raise ValueError('Invalid agent position')
			
		# Try to make the move given by agent_move
		new_loc = agent_loc.move(agent_move)
		
		# If the new location isn't valid, return the original location (bumped into a wall)
		if self.test_position_validity(new_loc):
			return new_loc
		else:
			return agent_loc
		
	def move_goal(self):
		if not self.test_position_validity(self.goal_loc):
			raise ValueError('Invalid agent position')
			
		# Try to make a random move
		goal_move = random.choice(self.possible_moves)
		new_loc = self.goal_loc.move(goal_move)
		
		# If the new position, is valid, move the goal; otherwise, keep it the same
		if self.test_position_validity(new_loc):
			self.goal_loc = new_loc
			
		# Return a copy of the goal location
		return self.goal_loc.clone()
	
	def test_position_validity(self,loc):
		pos = loc.coordinates()
		# Test if the location is within the grid dimensions
		if pos[0] < 0 or pos[0] > self.max_x or pos[1] < 0 or pos[1] > self.max_y:
			return False
		# Test if the location is an obstacle
		elif self.grid[pos] == self.obstacle:
			return False
		else:
			return True
		
	def visualize_world(self,agent_loc=None,V=None):
		# Fill obstacles with a black color
		obstacle_locs = np.argwhere(self.grid == self.obstacle)
		for loc in obstacle_locs:
			self.fill_plot_cell(loc,'k')
		# Fill agent location with blue and goal location with red
		if agent_loc is not None:
			self.fill_plot_cell(agent_loc.coordinates(),'b')
		self.fill_plot_cell(self.goal_loc.coordinates(),'r')
		plt.axis([0,self.max_x+1,0,self.max_y+1])
		if V is not None:
			self.draw_value_function(V)
		plt.xticks(np.arange(0,self.max_x+1))
		plt.yticks(np.arange(0,self.max_y+1))
		plt.grid()
		plt.show()
		
	def fill_plot_cell(self,loc,color):
		# Basic cell geometry
		xcell = np.array([0, 1, 1, 0]) + loc[0]
		ycell = np.array([0, 0, 1, 1]) + loc[1]
		plt.fill(xcell,ycell,color)
		
	def random_position(self):
		# Return a random position in the map
		invalid_position = True
		while invalid_position:
			pos = Position(np.random.randint(self.max_x+1),np.random.randint(self.max_y+1))
			invalid_position = not self.test_position_validity(pos)
			
		return pos
	
	def draw_value_function(self,V):
		x_surr = [-1, 1, 0, 0, 0]
		y_surr = [0, 0, 1, -1, 0]
		for x in range(0,self.max_x+1):
				for y in range(0,self.max_y+1):
					if not self.test_position_validity(Position(x,y)):
						continue
					if len(V.shape) == 2:
						action_vals = []
						for xi, yi in zip(x_surr,y_surr):
							if x+xi >= 0 and x+xi <= self.max_x and y+yi >= 0 and y+yi <= self.max_y:
								action_vals.append(V[x+xi,y+yi])
							else:
								action_vals.append(-1000)
					else:
						action_vals = V[x,y,:]
							
					dir_max = np.argmax(action_vals)
					if dir_max == 0: # Left
						dx = -0.5; dy = 0
						plt.arrow(x+0.75,y+0.5,dx,dy,width=0.05,length_includes_head=True)
					elif dir_max == 1: # Right
						dx = 0.5; dy = 0
						plt.arrow(x+0.25,y+0.5,dx,dy,width=0.05,length_includes_head=True)
					elif dir_max == 2: # Up
						dx = 0; dy = 0.5
						plt.arrow(x+0.5,y+0.25,dx,dy,width=0.05,length_includes_head=True)
					elif dir_max == 3: # Down
						dx = 0; dy = -0.5
						plt.arrow(x+0.5,y+0.75,dx,dy,width=0.05,length_includes_head=True)
					else: # Stay
						continue
							
					
	
if __name__ == "__main__":
	# Test the class by drawing the gridworld with correct initial locations
	gw = Gridworld(10,5,[(7,0),(7,1),(7,2)],(9,1))
	gw.visualize_world(Position(4,2))