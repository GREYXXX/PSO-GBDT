import random
import re
import numpy as np
from Tree import Tree
from GBDT import GBDTRegressor, GBDTMultiClassifier, GBDTBinaryClassifier
from loss import SquaresError, BinomialDeviance, MultinomialDeviance
from data import DataSet
from operator import attrgetter


"""
@author XI RAO
CITS4001 Research Project
"""

class Particle:

    def __init__(self, solution, fitness, id):

        """
        param solution : current potential solution of the particle, solution is a encoded gbdt array
        param fitness : fitness value (acc, rmse) of the current particle
        param id: particle id
        
        var pbest : best solution the particle has achieved so far
        var current_solution_fit : fitness value of the potential solution 
        var pbest_solution_fit : fitness value of the pbest solution 
        """

        self.solution = solution
        self.pbest = solution
        self.id = id

        self.current_solution_fit = fitness
        self.pbest_solution_fit = fitness

        self.velocity = []


    def set_pbest(self, new_pbest):
        """set pbest"""
        self.pbest =  new_pbest
    
    def get_pbest(self):
        """returns the pbest """
        return self.pbest

    def set_velocity(self, new_velocity):
         """set the new velocity (sequence of swap operators)"""
         self.velocity = new_velocity

    def get_velocity(self):
        """returns the velocity (sequence of swap operators)"""
        return self.velocity
    
    def set_current_solution(self, solution):
        """set current solution"""
        self.solution = solution

    def get_current_solution(self):
        """get current solution"""
        return self.solution

    def set_cost_pbest(self, fitness):
        """set fitness value for pbest solution"""
        self.pbest_solution_fit = fitness

    def get_cost_pbest(self):
        """gets fitness value of pbest solution"""
        return self.pbest_solution_fit

    def set_cost_current_solution(self, fitness):
        """set fitness value for the current solution"""
        self.current_solution_fit = fitness

    def get_cost_current_solution(self):
        """gets fitness value of the current solution"""
        return self.current_solution_fit

    def get_particle_id(self):
        """get particle's id"""
        return self.id

    def clear_velocity(self):
        """removes all elements of the list velocity"""
        del self.velocity[:]



class PSO():

    def __init__(self, dataset, iterations, size_population, max_tree_nums=5, learning_rate = 0.3, max_tree_depth = 4, 
                    model_type='regression', beta=1, alfa=1):

        self.dataset = dataset
        self.iterations = iterations
        self.size_population = size_population
        
        #The max tree numbers of GBDT
        self.max_tree_nums = max_tree_nums
        self.learning_rate = learning_rate
        self.max_tree_depth = max_tree_depth     #self.data.columns.size
        self.model_type = model_type
        self.particles = []
        self.beta = beta
        self.alfa = alfa

        if self.model_type == 'regression':
            self.gbdt = GBDTRegressor(self.learning_rate, self.max_tree_depth, self.max_tree_nums, SquaresError())
        elif self.model_type == 'binary_cf':
            self.gbdt = GBDTBinaryClassifier(self.learning_rate, self.max_tree_depth, self.max_tree_nums, BinomialDeviance())
        elif self.model_type == 'multi_cf':
            self.gbdt = GBDTMultiClassifier(self.learning_rate, self.max_tree_depth, self.max_tree_nums, MultinomialDeviance())
        else:
            raise ValueError("Invalid model type. Requires a valid model type: regression, binary_cf or multi_cf")

        #Init the particles
        self.init_swarm()
        
    
    def init_swarm(self):
        print("Particle initialization start....")
        for i in range(self.size_population):
            self.gbdt.build_gbdt(self.dataset, [])
            solution = self.gbdt.get_gbdt_array()
            self.particles.append(Particle(solution, self.get_fitness(self.gbdt), i))
            print("particle {} finished".format(i))
        print("Particle initialization finished")

    
    def get_fitness(self, gbdt):
        data = self.dataset.get_train_data()
        predict = gbdt.predict(data)

        if self.model_type == 'regression':
            #RMSE should be as small as possible, so negate the value make the fitness as big as possible
            fitness = - ((data['predict_value'] - data['label']) ** 2).mean() ** .5
        elif self.model_type == 'binary_cf':
            fitness = sum(predict['label'] == predict['predict_label']) / len(predict)
        elif self.model_type == 'multi_cf':
            fitness = sum(predict['label'] == predict['predict_label']) / len(predict)

        return fitness


    # set gbest (best particle of the population)
    def setGBest(self, new_gbest):
        self.gbest = new_gbest

    # returns gbest (best particle of the population)
    def get_gbest(self):
        return self.gbest

    def run(self):
        for iter in range(self.iterations):
            # updates gbest (best particle of the population)
            self.gbest = max(self.particles, key=attrgetter('pbest_solution_fit'))
            print("gbest is :{} at {} iter".format(self.gbest.get_cost_pbest(), iter))

            for par in self.particles:
                #calc fitness for each particle
                #get local best pbest -- the best fitness value in each particle's memory
                #get global best gbst -- the best fitness value accorss the whole population

                par.clear_velocity() # cleans the speed of the particle
                temp_velocity = []
                repeat_solutions = []
                solution_gbest = self.gbest.get_pbest() # gets solution of the gbest
                solution_pbest = par.get_pbest()[:] # gets the pbest solution
                solution_particle = par.get_current_solution()[:] # gets the current solution of the particle

                # generates all swap operators to calculate (pbest - x(t-1))
                for i in range(len(solution_particle)):
                    if solution_particle[i] != solution_pbest[i]:
                        # generates swap operator
                        swap_operator = (i, solution_pbest[i], self.alfa)

                        # append swap operator in the list of velocity
                        temp_velocity.append(swap_operator)

                # generates all swap operators to calculate (gbest - x(t-1))
                for i in range(len(solution_particle)):
                    if solution_particle[i] != solution_gbest[i]:
                        # generates swap operator
                        swap_operator = (i, solution_gbest[i], self.beta)

                        # append swap operator in the list of velocity
                        temp_velocity.append(swap_operator)

                
                par.set_velocity(temp_velocity)

                # generates new solution for particle
                for swap_operator in temp_velocity:
                    if random.random() <= swap_operator[2]:
                        # makes the swap
                        solution_particle[swap_operator[0]] = swap_operator[1]
                
                if len(repeat_solutions) == 0 or solution_particle not in repeat_solutions:
                    repeat_solutions.append(solution_particle)
                else:
                    print(f"repeated : {repeat_solutions}")
                    continue
                
                if self.model_type == 'regression':
                    self.gbdt = GBDTRegressor(self.learning_rate, self.max_tree_depth, self.max_tree_nums, SquaresError())
                elif self.model_type == 'binary_cf':
                    self.gbdt = GBDTBinaryClassifier(self.learning_rate, self.max_tree_depth, self.max_tree_nums, BinomialDeviance())
                elif self.model_type == 'multi_cf':
                    self.gbdt = GBDTMultiClassifier(self.learning_rate, self.max_tree_depth, self.max_tree_nums, MultinomialDeviance())
                else:
                    raise ValueError("Invalid model type. Requires a valid model type: regression, binary_cf or multi_cf")

                # updates the current solution
                par.set_current_solution(solution_particle)
                self.gbdt.build_gbdt(self.dataset, solution_particle)
                # gets cost of the current solution
                cost_current_solution = self.get_fitness(self.gbdt)
                # updates the cost of the current solution
                par.set_cost_current_solution(cost_current_solution)
                print(solution_particle, cost_current_solution, iter)

                # checks if current solution is pbest solution
                if cost_current_solution > par.get_cost_pbest():
                    par.set_pbest(solution_particle)
                    par.set_cost_pbest(cost_current_solution)

            print("\n")
    
		