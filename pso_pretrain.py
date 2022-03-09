
import numpy as np
import pandas as pd
import abc
import random
import pickle
from pip import main
import xgboost as xgb
from sklearn.model_selection import train_test_split
import pickle
import graphviz
from sklearn.tree import export_graphviz
from operator import attrgetter

import warnings
warnings.filterwarnings("ignore")



class Loss:
    
    def initialize_s_f_0(self, y):
        f0 = y.mean()
        return f0

    def initialize_b_f_0(self, y):
        pos = len(np.where(y == 1)[0])
        neg = y.shape[0] - pos
        f_0 = np.log(pos / neg)
        return f_0  
    

class Node:
    
    def __init__(self, feature, feature_val, id, remain_indexs, depth, parent, child, is_the_leaf = False):
        self.feature = feature
        self.feature_val = feature_val
        self.id = id
        self.remain_indexs = remain_indexs
        self.left = None
        self.right = None
        self.predict_value = None
        self.parent = parent
        self.depth = depth
        self.child = child
        
        if is_the_leaf:
            self.predict_value = feature_val

    def is_leaf(self):
        if self.left == None and self.right == None:
            return True
        else:
            return False

    def update_leaf_value(self, val):
        self.predict_value = val
    
    def predict(self, instance):
        if self.is_leaf():
            return self.predict_value
        if instance[self.feature] < self.feature_val:
            return self.left.predict(instance)
        else:
            return self.right.predict(instance)



class Tree:
    def __init__(self, tree_input, max_depth, tree_id):
        
        self.tree_input = tree_input
        self.max_depth = max_depth
        self.tree_id = tree_id
        self.id = 0
        self.leaf_nodes = []
        
        if len(tree_input) < 1:
            raise ValueError("Input tree is empty")
        else:
            feature = self.tree_input[0][0]
            feature_val = self.tree_input[0][1]
            child = self.tree_input[0][2]
            self.root = Node(feature, feature_val, id = 0 , depth = 0, parent = None, child = child)

    def __get_left_remain_indexs(self, df, feature, feature_value):
        if df[feature].dtype != 'object':
            return df[df[feature] < feature_value].index
        else:
            return df[df[feature] != feature_value].index
        
    
    def __get_right_remain_indexs(self, df, feature, feature_value):
        if df[feature].dtype != 'object':
            return df[df[feature] >= feature_value].index
        else:
            return df[df[feature] == feature_value].index

    def build_tree(self):
        queue = [self.root]
        while (len(queue) != 0):
            current_node = queue.pop(0)
            if current_node.depth < self.max_depth - 1:
                left_idx  = current_node.child[0]
                right_idx = current_node.child[1]
                if left_idx == '0' and right_idx == '0':
                    continue
                if current_node.left == None:
                    feature = self.tree_input[left_idx][0]
                    feature_val = self.tree_input[left_idx][1]
                    child = self.tree_input[left_idx][2]
                    if feature != 'Leaf':
                        current_node.left = Node(feature, feature_val, id, current_node.depth + 1, current_node, child)
                    else:
                        current_node.left = Node(feature, feature_val, id, current_node.depth + 1, 
                                                 current_node, child, is_the_leaf = True)
                    queue.append(current_node.left)

                if current_node.right == None:
                    feature = self.tree_input[right_idx][0]
                    feature_val = self.tree_input[right_idx][1]
                    child = self.tree_input[right_idx][2]
                    if feature != 'Leaf':
                        current_node.right = Node(feature, feature_val, id, current_node.depth + 1, current_node, child)
                    else:
                        current_node.right = Node(feature, feature_val, id, current_node.depth + 1, 
                                                  current_node, child, is_the_leaf = True)
                    queue.append(current_node.right)


    def rebuild_tree(self, tree_array, thresholds, data):
        
        if len(tree_array) < 1:
            raise ValueError("Input tree is empty")
        else:
            feature = tree_array[0][0][0]
            feature_val = tree_array[0][1]
            child = tree_array[0][2]
            self.root = Node(feature, feature_val, id = 0 , remain_indexs = data.index, depth = 0, parent = None, child = child)
        
        queue = []
        while (len(queue) != 0):
            current_node = queue.pop(0)
            current_data = data.loc[current_node.remain_indexs]
            if len(current_data) <= 1:
                current_node.update_leaf_value()

            if current_node.depth < self.max_depth - 1:
                left_child  = current_node.child[0]
                right_child = current_node.child[1]

                left_idxs = self.__get_left_remain_indexs(data, current_node.feature, current_node.feature_val)
                right_idxs = self.__get_right_remain_indexs(data, current_node.feature, current_node.feature_val)

                if current_node.left == None:
                    pass

                if current_node.right ==None:
                    pass
        pass                   
    
    def predict(self, features):
        return np.array([self.root.predict(row) for idx, row in features.iterrows()])
                


class GBDTInference:

    def __init__(self, learning_rate, max_depth, max_tree_nums, f_0):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_tree_nums = max_tree_nums
        self.f_0 = f_0
        self.trees = {}
        
    def build_gbdt(self, tree_array):
        for iter in range(self.max_tree_nums):
            tree = Tree(tree_array[iter], self.max_depth, iter)
            tree.build_tree()
            self.trees[iter] = tree

    def predict(self, x_val):
        f_pred = np.array([self.f_0 for i in range(len(x_val))], dtype = float)
        for iter in range(self.max_tree_nums):
            leafs = self.trees[iter].predict(x_val)
            f_pred += (self.learning_rate * leafs)
        
        condlist   = [1 / (1 + np.exp(-f_pred)) >= 0.5] 
        choicelist = [1]
        predict_label = np.select(condlist, choicelist, default = 0)

        return predict_label



class Particle:

    def __init__(self, solution, fitness, id):
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

    def __init__(self, feature_vals, min_maxs, iterations, size_population, w, c1, c2, f_0, model):
        """
        param min_maxs : (min, max) for each input feature values (feature_vals)
        """

        self.feature_vals = feature_vals
        self.min_maxs = min_maxs
        self.iterations = iterations
        self.size_population = size_population
        self.w = w
        self.c1 = c1
        self.c2 = c2
        

        self.max_tree_nums = model.get_n_trees()
        self.learning_rate = model.get_learning_rate()
        self.max_tree_depth = model.get_max_depth() 
        self.tree_array = model.get_tree_array()
        self.particles = []

        if len(feature_vals) != min_maxs:
            raise ValueError("min_maxs array should have the same lengths as the input feature values")
        
        self.gbdt = GBDTInference(
            learning_rate = self.learning_rate, 
            ax_depth = self.max_tree_depth, 
            max_tree_nums = self.max_tree_nums, 
            f_0 = f_0
        )
        
        #Init the particles
        self.init_swarm()
        
    
    def init_swarm(self):
        print("Particle initialization start....")
        solution = [(random.random() * (i[1] - i[0]) + i[0]) for i in self.min_maxs]
        velocity = [random.random() for i in range(len(self.min_maxs))]
        for i in range(self.size_population):
            self.gbdt.build_gbdt(self.tree_array, solution)
            par = Particle(solution, self.get_fitness(self.gbdt), i)
            par.set_velocity(velocity)
            self.particles.append(par)
            print("particle {} finished".format(i))
        print("Particle initialization finished")

    
    def get_fitness(self, gbdt):
        pass


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

                par.clear_velocity() # cleans the speed of the particle
                temp_velocity = []
                repeat_solutions = []
                solution_gbest = self.gbest.get_pbest() # gets solution of the gbest
                solution_pbest = par.get_pbest()[:] # gets the pbest solution
                solution_particle = par.get_current_solution()[:] # gets the current solution of the particle
                pass



if __name__ == "__main__":
    # df = pd.read_csv("data/BankNote.csv")
    # X = df.drop("class", axis=1)
    # y = df["class"]

    df = pd.read_csv("data/classification.csv")
    X = df.drop("success", axis=1)
    y = df["success"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    xgbd = xgb.XGBClassifier(n_estimators=6, learning_rate=1, max_depth=4)
    xgbd.fit(X_train, y_train)

    pickle.dump(xgbd, open("xgb.pkl", "wb"))

    m = PreprocessModel("xgb.pkl")
    arrays = m.get_tree_arrays()
    loss = Loss()
    f_0  = loss.initialize_b_f_0(y_val)
    gbdt = GBDTInference(m.get_learning_rate(), m.get_max_depth() + 1, m.get_n_trees(), f_0)
    gbdt.build_gbdt(arrays)
    predict = gbdt.predict(X_val)
    acc = len(np.where(predict == y_val)[0]) / len(X_val)
    print(f"re-build acc is {acc}")
    print('\n')
    print(m.get_model().score(X_val, y_val))
