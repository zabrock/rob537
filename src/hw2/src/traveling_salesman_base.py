#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 21:00:46 2019

@author: Zeke
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:04:58 2019

@author: Zeke
"""
import numpy as np

class TravelingSalesmanBase(object):
	def __init__(self,x,y):
		self.x = x
		self.y = y
		self.calc_cost_matrix()
		
	def calc_cost_matrix(self):
		# Precalculate the costs to travel between each city
		self.costs = np.zeros((len(self.x),len(self.x)))
		for i in range(0,len(self.x)):
			for j in range(i+1,len(self.x)):
				x_diff = self.x[i] - self.x[j]
				y_diff = self.y[i] - self.y[j]
				dist = np.sqrt(x_diff**2 + y_diff**2)
				self.costs[i][j], self.costs[j][i] = dist, dist
		
	def calc_cost(self, state):
		cost = 0
		for i in range(0,len(state)-1):
			cost += self.costs[state[i]][state[i+1]]
			
		return cost