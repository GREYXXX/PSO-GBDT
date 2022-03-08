import queue
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
import time
from data import DataSet
from GBDT import GBDTBinaryClassifier, GBDTRegressor
from pso import PSO
from loss import BinomialDeviance, SquaresError



import warnings
warnings.filterwarnings("ignore")


class PreprocessModel:
    def __init__(self, file):
        self.model = pickle.load(open(file, "rb"))
        self.n_trees = self.model.n_estimators
        self.max_depth = self.model.max_depth
        self.learning_rate = self.model.learning_rate
        self.tree_df = self.model.get_booster().trees_to_dataframe()
        self.internal_nodes = [(self.tree_df['Feature'].values[i], self.tree_df['Split'].values[i]) 
                                    for i in range(len(self.tree_df)) if self.tree_df['Feature'].values[i] != 'Leaf']
    
    def get_model(self):
        return self.model 

    def get_n_trees(self):
        return self.n_trees
    
    def get_max_depth(self):
        return self.max_depth 
    
    def get_learning_rate(self):
        return self.learning_rate
    
    def get_internal_splits(self):
        return self.internal_nodes
    


def pso_run(df, pretrain_file, target_name):

    df = df.sample(frac=1).reset_index(drop=True)
    train = df[:int(df.shape[0] * 0.8)]
    test  = df[int(df.shape[0] * 0.8):]

    model = PreprocessModel(pretrain_file)
    internal_splits = model.get_internal_splits()

    dataset = DataSet(train, test, target_name, standardize=False)
    print("encode start...")
    dataset.encode_table(pretrained=True, pretrain_arrays=internal_splits)
    print("encode done...")
    lookup_tables = dataset.get_lookup_table()

    pretrain_arrays = [dataset.get_index(val) for val in internal_splits]
    iterations=20
    size_population=500
    max_tree_nums=6
    learning_rate = 1
    max_tree_depth = 6

    start = time.time()
    pso = PSO(
        dataset, 
        iterations=iterations, 
        size_population=size_population,  
        pretrain_nodes=pretrain_arrays,
        pretrained=True,
        max_tree_nums=max_tree_nums, 
        learning_rate = learning_rate, 
        max_tree_depth = max_tree_depth, 
        model_type='binary_cf', 
        beta=0.3, 
        alfa=0.3
    )
    pso.run() # runs the PSO algorithm
    print('gbest: %s | cost: %f\n' % (pso.get_gbest().get_pbest(), pso.get_gbest().get_cost_pbest()))

    end = time.time()

    #Build the gbdt with gbest solution
    gbdt = GBDTBinaryClassifier(
        learning_rate=learning_rate, 
        max_depth=max_tree_depth, 
        max_tree_nums=max_tree_nums, 
        loss = BinomialDeviance()
    )
    gbdt.build_gbdt(dataset, pso.get_gbest().get_pbest())
    #Predict the results for test dataset
    predict = gbdt.predict(dataset.get_test_data())
    #print the accuracy
    print(sum(predict['label'] == predict['predict_label']) / len(predict))
    print(f"time taken is {end - start}")


if __name__ == "__main__":

    df = pd.read_csv("/Users/xirao/data/covat_0.3.csv")
    df = df.drop('Unnamed: 0', axis = 1)
    df['54'] = df['54'].apply(lambda x : 1 if x > 1.0 else 0)
    
    # train_file = 'data/BankNote.csv'
    # pretrain_file = '/Users/xirao/data/xgb.pkl'
    # target_name = 'class'

    pretrain_file = 'pretrain_models/covat_0.3.pkl'
    target_name = '54'
    pso_run(df, pretrain_file, target_name)