# @Author XI RAO
# CITS4001 Research Project

import random
from tkinter import NONE
import numpy as np
from Tree import Node, Tree
from GBDT import GBDTRegressor, GBDTMultiClassifier, GBDTBinaryClassifier
from loss import SquaresError, BinomialDeviance, MultinomialDeviance
from data import DataSet
from operator import attrgetter
import pandas as pd
from typing import List, Tuple



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



class PSO:

    def __init__(
        self,  
        iterations : int, 
        size_population : int, 
        max_tree_nums : int = 5, 
        learning_rate : int = 0.3, 
        max_tree_depth : int = 4, 
        model_type : str = 'regression', 
        beta : float = 0.45, 
        alfa : float = 0.45
    ):

        self.iterations = iterations
        self.size_population = size_population
        
        self.max_tree_nums = max_tree_nums
        self.learning_rate = learning_rate
        self.max_tree_depth = max_tree_depth 
        self.model_type = model_type
        self.particles = []
        self.gbest_record = []
        self.beta = beta
        self.alfa = alfa

        if self.model_type == 'regression':
            self.gbdt = GBDTRegressor(
                self.learning_rate, 
                self.max_tree_depth, 
                self.max_tree_nums, 
                SquaresError()
                )

        elif self.model_type == 'binary_cf':
            self.gbdt = GBDTBinaryClassifier(
                self.learning_rate, 
                self.max_tree_depth, 
                self.max_tree_nums, 
                BinomialDeviance()
                )
                
        elif self.model_type == 'multi_cf':
            self.gbdt = GBDTMultiClassifier(
                self.learning_rate, 
                self.max_tree_depth, 
                self.max_tree_nums, 
                MultinomialDeviance()
                )
        else:
            raise ValueError("Invalid model type. Requires a valid model type: regression, binary_cf or multi_cf")
       
    def init_swarm(self, dataset : DataSet) -> None:
        print("Train without pretrain, and Particle initialization start....")
        for i in range(self.size_population):
            self.gbdt.build_gbdt(dataset = dataset)
            solution = self.gbdt.get_gbdt_array()
            self.particles.append(Particle(solution, self.get_fitness(dataset=dataset), i))
            print("particle {} finished".format(i))
        print("Particle initialization finished")

    def init_swarm_with_nodes(self, dataset : DataSet, pretrain_nodes : List[int])-> None:
        print("Train with pretrain, and Particle initialization start....")
        for i in range(self.size_population):
            sample_nodes = random.choices(
                pretrain_nodes, 
                k = self.max_tree_depth * self.max_tree_nums
            )
            self.gbdt.build_gbdt(dataset = dataset, tree_array = sample_nodes)
            solution = self.gbdt.get_gbdt_array()
            self.particles.append(Particle(solution, self.get_fitness(dataset=dataset), i))
            print("particle {} finished".format(i))
        print("Particle initialization finished")
    
    def get_fitness(self, dataset : DataSet) -> float:
        train_data = dataset.get_train_data()
        predict = self.gbdt.predict(train_data)

        if self.model_type == 'regression':
            #RMSE should be as small as possible, so negate the value make the fitness as big as possible
            fitness = - ((predict['predict_value'] - predict['label']) ** 2).mean() ** .5
        elif self.model_type == 'binary_cf':
            fitness = sum(predict['label'] == predict['predict_label']) / len(predict)
        elif self.model_type == 'multi_cf':
            fitness = sum(predict['label'] == predict['predict_label']) / len(predict)

        return fitness

    # set gbest (best particle of the population)
    def setGBest(self, new_gbest):
        self.gbest = new_gbest

    def run(
        self, 
        X : pd.DataFrame, 
        y : pd.Series, 
        internal_splits : List[Tuple[str, float]] = [],
        use_pretrain : bool = False
        )-> None:

        # Encoding dataset
        dataset = DataSet(X, y)
        if not use_pretrain:
            dataset.encode_table()
        else:
            dataset.encode_table(use_pretrain = use_pretrain, internal_splits = internal_splits)

        # Init particles
        if use_pretrain:
            pretrain_discrete_arrays = [dataset.get_index(val) for val in internal_splits]
            self.init_swarm_with_nodes(dataset = dataset, pretrain_nodes = pretrain_discrete_arrays)
        else:
            self.init_swarm(dataset = dataset)

        # record_fit = []

        #Init the particles
        for iter in range(self.iterations):
            # updates gbest (best particle of the population)
            self.gbest = max(self.particles, key=attrgetter('pbest_solution_fit'))
            self.gbest_record.append(self.gbest.get_cost_pbest())
            print("gbest is :{} at {} iter".format(self.gbest.get_cost_pbest(), iter))
            
            # if len(record_fit) != 0:
            #     per = record_fit.count(max(set(record_fit), key = record_fit.count)) / len(record_fit)
            #     print(per)
            #     if per > .8:
            #         print(f"early stop trigger")
            #         break
                    
            #     record_fit.clear()

            for par in self.particles:
                par.clear_velocity() # cleans the speed of the particle
                temp_velocity = []
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

                # updates the current solution
                par.set_current_solution(solution_particle)
                self.gbdt.build_gbdt(dataset, solution_particle)

                # gets cost of the current solution
                cost_current_solution = round(self.get_fitness(dataset), 5)

                # record_fit.append(cost_current_solution)
                # updates the cost of the current solution
                par.set_cost_current_solution(cost_current_solution)
                print(solution_particle, cost_current_solution, iter)

                # checks if current solution is pbest solution
                if cost_current_solution > par.get_cost_pbest():
                    par.set_pbest(solution_particle)
                    par.set_cost_pbest(cost_current_solution)

            print("\n")
        self.dataset_obj = dataset

    # returns gbest (best particle of the population)
    def get_gbest(self):
        return self.gbest
    
    def get_gbest_records(self):
        return self.gbest_record

    @property
    def dataset(self):
        return self.dataset_obj
		