#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:20:43 2019

@author: Zeke
"""

import matplotlib.pyplot as plt
import numpy as np

class VisualizeTS(object):
	def __init__(self,x,y):
#		self.fig, self.ax = plt.subplots()
		self.x = x
		self.y = y
		
	def plot_solution(self,order):
#		self.ax.cla()
		for pt_x, pt_y in zip(self.x, self.y):
			plt.plot(pt_x, pt_y, 'bo')
		augmented_order = np.append(order,order[0])
		x_lines = [self.x[idx] for idx in augmented_order]
		y_lines = [self.y[idx] for idx in augmented_order]
		plt.plot(x_lines,y_lines,'r')
		plt.show()
			