#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 00:09:05 2021

@author: suenchihang
"""

import random
import numpy as np
from math import comb
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats.distributions
import itertools


random.seed(1)
rng = np.random.default_rng()

#random functions
def uniform_integer (mean=4.5, variance=8.25):
    return np.random.randint(0, mean*2+1)

def uniform_continuous(mean=4.5, variance=6.75):
    return np.random.random()*mean*2

def poisson (mean=4.5, variance=4.5):
    return rng.poisson(mean)

def negative_binomial (k=3, p=2/3): #default has mean=4.5
    return rng.negative_binomial(k, p)

def gamma (alpha, scale):
    return rng.gamma(alpha, scale)

def lognormal (mean, sigma):
    return rng.lognormal(mean, sigma)

#L-shape path for illustration, grid_environment using 2-D array
def Lshape_path(grid_2Darray):
    L_path = np.zeros_like(grid_2Darray)
    L_path[1:, 0] = 1
    L_path[-1, 1:-1] = 1
    time = np.sum(grid_2Darray * L_path)
    return L_path, time



def naive_path(grid_2Darray):
    """
    All combinations of right and down are calculated to get the minimum path, 
    with total no. of steps = (no. of rows -1) + (no. of columns -1). 
    """
    rows, columns = np.shape(grid_2Darray)
    
    if rows == 1 or columns == 1:   #check for trivial case
        n_path = np.ones_like(grid_2Darray)
        n_path[0,0] = 0
        n_path[-1,-1] = 0
        best_time = np.sum(grid_2Darray * n_path)
    else:
    
        n_path = np.zeros_like(grid_2Darray)

        right = [0,1]
        down = [1,0]
        steps = rows+columns-2
        path_all_right = np.tile(right, (steps, 1))
        #initialize with maximum value
        best_time = np.max(grid_2Darray)*(rows+columns-2)
    
        #loop for combinations from moves of (rows-1) down and (columns-1) right
        for i in itertools.combinations(range(steps), rows-1):
            path_steps = path_all_right.copy()
            position = [0,0]
            path_map = np.zeros((rows, columns))
        
            for x in i: #generate each path steps
                path_steps[x] = down
            
            for j in range(steps-1): #exclude destination step, generate path map data
                position = position + path_steps[j]
                n, m = position
                path_map[n, m] = 1 
            
            time = np.sum(grid_2Darray * path_map) 
            if time <= best_time:
                n_path = path_map.copy() 
                best_time = time
        
    return n_path, best_time


"""
Naïve_split_path below is created by splitting the grid into two triangles along a diagonal, 
as the path must pass through at least one cell of the diagonal, 
which is each treated as end-point of first part and start-point of the second part.  
Run the naïve algorithm for each diagonal cell twice for each part 
(which is much smaller than the whole triangle) and sum up the result, 
then we can choose the diagonal cell through which the combined path is the shortest. 
For non-square grid, shorter side is used to choose “diagonal” cells. 
"""
def naive_split_path(grid_2Darray):
    rows, columns = np.shape(grid_2Darray)
    
    if comb(rows+columns-2, rows-1) < 1000:
        n_path, best_time = naive_path(grid_2Darray)
    else:
        best_time = np.max(grid_2Darray)*(rows+columns-2)  #initiate with a large num
        n_path = np.zeros((rows, columns))
        #split by checking diagonal points
        for i in range(min(rows,columns)):
            j = columns-1-i
            n_path1, best_time1 = naive_path(grid_2Darray[0:(i+1), 0:(j+1)])
            n_path2, best_time2 = naive_path(grid_2Darray[i:rows, j:columns])
            best_time_temp = best_time1 + best_time2 + grid_2Darray[i,j]
            if best_time_temp < best_time:
                best_time = best_time_temp
                n_path1 = np.concatenate((n_path1, np.zeros((rows-1-i,j+1))),axis=0)
                n_path1 = np.concatenate((n_path1, np.zeros((rows,columns-j-1)) ),axis=1)
                n_path2 = np.concatenate((np.zeros((i,columns-j)), n_path2),axis=0 )
                n_path2 = np.concatenate((np.zeros((rows,j)), n_path2),axis=1)
                n_path = n_path1 + n_path2
                n_path[i,j] = 1
                
    return n_path, best_time


def Dijkstra(grid_2Darray):
    rows, columns = np.shape(grid_2Darray)
    if rows == 1 or columns == 1: #solve for trivial case
        n_path = np.ones_like(grid_2Darray)
        n_path[0,0] = 0
        n_path[-1,-1] = 0
        best_time = np.sum(grid_2Darray * n_path)
    else:
        #initialization
        n_path = np.zeros_like(grid_2Darray)
        Max = np.amax(grid_2Darray)*rows*columns #for later initializiation with infinity
        visit = np.zeros_like(grid_2Darray) #1 for visited, 0 for unvisited
        distance = np.full((rows, columns),Max) #initialize each node as very large distance (aka infinity in original version) for comparing in later stage
        distance[0,0]=0 # first current node
        preceding_node_set = np.empty((rows, columns), dtype=object)
        current_node = np.array([0,0],dtype=int)
        U = [-1,0]
        D = [1, 0]
        L = [0,-1]
        R = [0,1]
        actions = [U, D, L, R]
       
        #loop until destination is visited
        while visit[-1,-1] == 0 :
            for action in actions:
                i,j = current_node + action #look for neighbour [i,j]
                if i in range(0,rows) and j in range(0,columns) and visit[i,j] == 0:
                    distance_temp = distance[current_node[0], current_node[1]] + grid_2Darray[i,j]
                    if distance_temp < distance[i,j]:
                        distance[i,j] = distance_temp
                        preceding_node_set[i,j] = current_node
            visit[current_node[0],current_node[1]] = 1 #update list of visited nodes so that current node won't be visited again
            distance[current_node[0],current_node[1]] = float('inf')  #equivalent to removing it for checking shortest distance node that is not yet visited
            current_node = np.asarray(divmod(np.nanargmin(distance),columns)) #get node of shortest distance
        
        #get path
        i,j = preceding_node_set[-1,-1]
        while not(i==0) or not(j==0) :
            n_path[i,j] += 1
            i,j = preceding_node_set[i,j]
        
        best_time = np.sum(grid_2Darray * n_path)
    return n_path, best_time


#set the rectangular grid
class grid:
    
    def __init__ (self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.data = np.zeros((rows, columns))
        self.df = pd.DataFrame(self.data)
        
    def populate (self, arg1=4.5, arg2=8.25, random_function=uniform_integer):
        with np.nditer(self.data, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = random_function(arg1, arg2)
        self.df = pd.DataFrame(self.data)
        
   # def populate_manual(self, 2Darray):
    #    self.data = 2Darray
      #  self.df = pd.DataFrame(self.data)
        
    def display (self):
        print("The grid of the game is as follows. Pls find the fastest path from top left corner to bottom right corner.")
        display(self.df)
        

class path():
    def __init__ (self, grid_2Darray):
        self.grid = grid_2Darray
        self.rows, self.columns = np.shape(self.grid)
     
        
    def populate(self, path_function = Dijkstra):
        self.data, self.time = path_function(self.grid)
        self.df = pd.DataFrame(self.data)
        self.algorithm = path_function.__name__
        
    
    def display (self, style = 3):
        print("")
        print("Using "+self.algorithm+" algorithm, the time cost is "+str(self.time))
        
        if style == 1 or style =="all":
            print("The path is in green as follows:")
            for i in range(self.rows):
                for j in range(self.columns):
                    if self.data[i,j] == 0:
                        print(str(self.grid[i, j]).rjust(6), end = " ")
                    else:
                        print(("\033[1;32;40m" + str(self.grid[i, j]).rjust(6) + "\033[0m"), end = " ")
                print()
        
        if style == 2 or style =="all":
            print("The fastest path excluding end-points shown in simple map:")
            plt.imshow(self.data, cmap="hot", interpolation='nearest')
            plt.show()
        
        
        if style == 3 or style == "all":
            #reference: Matplotlib website on creating annotated heatmap
            fig, ax = plt.subplots()
            im = ax.imshow(self.data)
            ax.set_xticks(np.arange(self.columns))
            ax.set_yticks(np.arange(self.rows))
            ax.set_xticklabels(np.arange(self.columns))
            ax.set_yticklabels(np.arange(self.rows))
            for i in range(self.rows):
                for j in range(self.columns):
                    text = ax.text(j, i, self.grid[i, j], ha="center", va="center", color="g")
            ax.set_title("The fastest path excluding end-points")
            fig.tight_layout()
            plt.show()


def simulation(rows=10, columns=10, arg1=4.5, arg2=8.25, random_function = uniform_integer , path_function = Dijkstra, sample=1000):
        game_test=grid(rows, columns)
        path_list = []
        for i in range(sample):
            game_test.populate(arg1, arg2, random_function)
            path_test = path(game_test.data)
            path_test.populate(path_function)
            path_list.append(path_test.time)
        
        path_values = np.array(path_list)
        path_mean = np.sum(path_values)/sample
        sample_variance = np.var(path_values) * sample/(sample-1)
            
        return path_mean, sample_variance, path_values
     

def size_factor(start=4, end=30, mean=10, sample=1000): #vary the length of grid, holding mean of each cell constant
    mean_list = []
    variance_list = []
    for i in range(start, end+1):
        path_mean, sample_variance, path_values = simulation(i,i,mean,sample=sample)
        mean_list.append(path_mean)
        variance_list.append(sample_variance)
    
    standard_deviation = [j**0.5 for j in variance_list]
    x = np.arange(start, end+1, 1)
    
    plt.plot(x, mean_list, label="Path mean")
    plt.plot(x, standard_deviation, label="Path standard deviation", linestyle='dotted')
    plt.plot(x, variance_list, label="Path sample variance")
    plt.title("How the size of grid affects the shortest path")
    plt.xlabel("Length of square grid")
    plt.legend()
    plt.show()
        

def shape_factor(): #hold grid size unchanged at 144 cells, use uniform integer distribution of mean 10 for simulation
    rows = [1,2,3,4,6,12,24,36,48,72,144]
    columns = [int(rows[-1]/i) for i in rows] 
    mean_list = []
    variance_list = []
    for i in range(len(rows) ):
        path_mean, sample_variance, path_values = simulation(rows[i],columns[i],10)
        mean_list.append(path_mean)
        variance_list.append(sample_variance)
    
    standard_deviation = [j**0.5 for j in variance_list]
    plt.plot(rows, mean_list, label="Path mean")
    plt.plot(rows, standard_deviation, label="Path standard deviation", linestyle='dotted')
    plt.plot(rows, variance_list, label="Path sample variance")
    plt.title("How the shape of grid affects the shortest path")
    plt.xlabel("Length of rectangular grid of same area 144 cells")
    plt.legend()
    plt.show()

def discrete_continuous(start_mean=4, end_mean=20, length=10): #vary mean of distribution, holding size of grid constant
    mean_list1 = []
    variance_list1 = []
    mean_list2 = []
    variance_list2 = []
    for i in range(start_mean, end_mean+1):
        path_mean1, sample_variance1, path_values1 = simulation(length,length,i)
        mean_list1.append(path_mean1)
        variance_list1.append(sample_variance1)
        path_mean2, sample_variance2, path_values2 = simulation(length,length,i, random_function=uniform_continuous)
        mean_list2.append(path_mean2)
        variance_list2.append(sample_variance2)
    
    x = np.arange(start_mean, end_mean+1, 1)
    
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(x, mean_list1, label = "uniform discrete")
    axis[0].plot(x, mean_list2, label = "uniform continuous")
    axis[0].set_title("Path mean")
    axis[1].plot(x, variance_list1, label = "uniform discrete")
    axis[1].plot(x, variance_list2, label = "uniform continuous")
    axis[1].set_title("Path variance")
    plt.xlabel("Mean of each cell")
    plt.suptitle("Path varying with mean of uniform discrete and continuous distribution")
    plt.legend()
    plt.show() 
    

def distribution_factor(start, end, step=1, arg1=float('inf'), arg2=float('inf'),  length=10, distribution=poisson, xlabel=""): #vary parameter of distribution, holding size of grid constant
    mean_list1 = []
    variance_list1 = []
    x = np.arange(start, end, step)
    #check and assign which argument to loop and which remains constant
    if arg1==float('inf'):
        arg1_array = x
        arg2_array = np.array([arg2])
    else:
        arg2_array = x
        arg1_array = np.array([arg1])
    
    #loop to get different values of mean and variance using simulation function
    for i in arg1_array:
        for j in arg2_array:
            path_mean1, sample_variance1, path_values1 = simulation(length,length, i, j, random_function=distribution)
            mean_list1.append(path_mean1)
            variance_list1.append(sample_variance1)
    
    #plot the charts for path mean and variance
    figure, axis = plt.subplots(2, 1)
    axis[0].plot(x, mean_list1, label="Path mean")
    axis[0].legend(loc="upper left")
    axis[1].plot(x, variance_list1, label="Path variance")
    axis[1].legend(loc="upper left")
    plt.xlabel(xlabel)
    plt.suptitle("Path varying with "+distribution.__name__+" distribution "+ xlabel)
    plt.tight_layout()
    plt.show() 


game1 = grid(10, 10)
game1.populate(10)
game1.display()
game1_path = path(game1.data)



start_time2=time.time()
game1_path.populate(naive_split_path)
time2=time.time()-start_time2
game1_path.display()
print("Computation time by naive_split_path: "+str(time2))

start_time3=time.time()
game1_path.populate(Dijkstra)
time3=time.time()-start_time3
game1_path.display(style=3)
print("Computation time by Dijkstra: "+str(time3))



size_factor(4,30, mean=10)
shape_factor()
discrete_continuous(4,20, length=10)
distribution_factor(4, 21, distribution=poisson, xlabel="λ" )
distribution_factor(4, 21, arg2=0.5, distribution=negative_binomial, xlabel="k" ) #k varies, holding the same p=0.5
distribution_factor(0.05, 1, step=0.05, arg1=4, distribution=negative_binomial, xlabel="p" ) #p varies, holding the same k=4
distribution_factor(4, 21, arg2=0.5, distribution=gamma, xlabel="α" ) #α varies, holding the same λ=0.5 (and this becomes Chi-square distribution)
distribution_factor(0.5, 10.5, step=0.5, arg1=4, distribution=gamma, xlabel="λ" ) #λ varies, holding the same α=4
distribution_factor(4, 21, arg2=1, distribution=lognormal, xlabel="μ" ) #μ varies, holding the same σ=1
distribution_factor(0.5, 10.5, step=0.5, arg1=4, distribution=lognormal, xlabel="σ" ) #λ varies, holding the same μ=4

        
    