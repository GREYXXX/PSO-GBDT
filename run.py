
from data import DataSet
from loss import SquaresError, BinomialDeviance, MultinomialDeviance
from GBDT import GBDTRegressor, GBDTBinaryClassifier, GBDTMultiClassifier
import pandas as pd
from data import DataSet
import random
from pso import PSO
import xgboost as xgb
from xgboost import plot_tree
from catboost import CatBoostClassifier
import os
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class BasePSORunner(object):

    def __init__(self, file, target_name, is_bin = False, standardize=False):
        self.df = pd.read_csv(file)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.train = self.df[:int(self.df.shape[0] * 0.8)]
        self.test  = self.df[int(self.df.shape[0] * 0.8):]
        self.is_bin = is_bin

        is_bin = False
        dataset = DataSet(self.train, self.test, target_name, standardize=standardize, is_bin=is_bin)

        print("encode start...")
        dataset.encode_table()
        print("encode done...")
        self.lookup_tables = dataset.get_lookup_table()
    
    @property
    def get_lookup_tables(self):
        return self.lookup_tables
    
    @staticmethod
    def plot_pso(x, y, name):
        plt.grid(which='major', axis='y', color='darkgray')
        plt.grid(which='major', axis='x', color='darkgray')
        plt.plot(x,y,color='fuchsia', marker='P', linestyle='dashed')
        plt.xlabel('Number of iteration',size = 15)
        plt.ylabel('Fitness value',size = 15)
        plt.title('PSO with pure search')
        # plt.show() 
        plt.savefig(name)


class PSOBinaryClassifierRunner(BasePSORunner):
    
    def __init__(self, file, target_name, is_bin=False, standardize=False):
        super().__init__(file, target_name, is_bin, standardize)

    def run():
        pass

class PSORegressorRunner(BasePSORunner):

    def __init__(self, file, target_name, is_bin=False, standardize=False):
        super().__init__(file, target_name, is_bin, standardize)

    def run():
        pass


def pso_run(
    name, 
    dataset, 
    iterations = 20, 
    size_population = 50, 
    max_tree_nums=6, 
    learning_rate = 1, 
    max_tree_depth = 5, 
    model_type = 'binary_cf',
    alpha = 0.45,
    beta = 0.45,
    is_bin = False
    ):

    start = time.time()
    pso = PSO(
        dataset, 
        iterations=iterations, 
        size_population=size_population,  
        max_tree_nums=max_tree_nums, 
        learning_rate = learning_rate, 
        max_tree_depth = max_tree_depth, 
        model_type = model_type, 
        beta = beta, 
        alfa = alpha
    )
    pso.run() # runs the PSO algorithm
    print('gbest: %s | cost: %f\n' % (pso.get_gbest().get_pbest(), pso.get_gbest().get_cost_pbest()))
    end = time.time()

    if model_type == 'binary_cf':

        # Build the gbdt binary classifier with gbest solution
        gbdt = GBDTBinaryClassifier(
            learning_rate=learning_rate, 
            max_depth=max_tree_depth, 
            max_tree_nums=max_tree_nums, 
            loss = BinomialDeviance()
        )
    elif model_type == 'regression':

        # Build the gbdt regressor with gbest solution
        gbdt = GBDTRegressor(
            learning_rate=learning_rate, 
            max_depth=max_tree_depth, 
            max_tree_nums=max_tree_nums, 
            loss = SquaresError()
        )
    
    elif model_type == 'multi_cf':

        # Build the gbdt regressor with gbest solution
        gbdt = GBDTMultiClassifier(
            learning_rate=learning_rate, 
            max_depth=max_tree_depth, 
            max_tree_nums=max_tree_nums, 
            loss = MultinomialDeviance()
        )
    else:
        raise ValueError("Please input a valid model type (binary_cf, regression or multi_cf)")

    gbdt.build_gbdt(dataset, pso.get_gbest().get_pbest())

    # Predict the results for test dataset
    predict = gbdt.predict(dataset.get_test_data())

    # print the accuracy
    print(sum(predict['label'] == predict['predict_label']) / len(predict))
    print(f"time taken is {end - start}")

    y = pso.get_gbest_records()
    x = np.arange(iterations)

    base = 'pso-search'
    if is_bin:
        name = base + '-' + name + '-bins' + '-p' + str(size_population) + '-i' + str(iterations) 
    else:
        name = base + '-' + name + '-p' + str(size_population) + '-i' + str(iterations) 
    print(name)

    pickle.dump((pso.get_gbest().get_pbest()), open('train_models/' + name + '.pkl', "wb"))
    plot_pso(x, y, 'images/' + name + '.svg')




