from data import DataSet
from loss import SquaresError, BinomialDeviance, MultinomialDeviance
from GBDT import GBDTRegressor, GBDTBinaryClassifier, GBDTMultiClassifier
import pandas as pd
from data import DataSet
import random
from pso import PSO
import xgboost as xgb
from xgboost import plot_tree
from catboost import CatBoostRegressor
import os
import pickle
import time
import re

from sklearn.model_selection import train_test_split
from sklearn import tree as sktree
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
from ProcessModel import PreprocessXgbModel, PreprocessSklModel

import warnings
warnings.filterwarnings("ignore")


# df = pd.read_csv('/Users/xirao/data/kc_house_data.csv')
# label = df['price']
# df = df.drop('id',axis=1)
# df = df.drop('date', axis=1)
# df = df.drop('price', axis=1)
# df['label'] = label
# target_name = 'label'

# df = pd.read_csv('/Users/xirao/data/insurance.csv')
# df['smoker'] = pd.factorize(df['smoker'])[0]
# df['sex']    = pd.factorize(df['sex'])[0]
# df['region'] = pd.factorize(df['smoker'])[0]
# target_name = 'charges'

df = pd.read_csv('/Users/xirao/data/winequality-red.csv')
target_name = 'quality'

df = df.sample(frac=1).reset_index(drop=True)
train = df[:int(df.shape[0] * 0.8)]
test  = df[int(df.shape[0] * 0.8):]

is_bin = True
dataset = DataSet(train, test, target_name, standardize=False, is_bin=is_bin)
print("encode start...")
dataset.encode_table()
print("encode done...")
lookup_tables = dataset.get_lookup_table()


def plot_pso(x, y, name):
    plt.grid(which='major', axis='y', color='darkgray')
    plt.grid(which='major', axis='x', color='darkgray')
    plt.plot(x,y,color='fuchsia', marker='P', linestyle='dashed')
    plt.xlabel('Number of iteration',size = 15)
    plt.ylabel('Fitness value',size = 15)
    plt.title('PSO with pure search')
    # plt.show() 
    plt.savefig(name)


def pso_run(name):

    iterations=20
    size_population=50
    max_tree_nums=6
    learning_rate = 1
    max_tree_depth = 5

    start = time.time()
    pso = PSO(
        dataset, 
        iterations=iterations, 
        size_population=size_population,  
        max_tree_nums=max_tree_nums, 
        learning_rate = learning_rate, 
        max_tree_depth = max_tree_depth, 
        model_type='regression', 
        beta=0.45, 
        alfa=0.45
    )
    pso.run() # runs the PSO algorithm
    print('gbest: %s | cost: %f\n' % (pso.get_gbest().get_pbest(), pso.get_gbest().get_cost_pbest()))

    end = time.time()

    #Build the gbdt with gbest solution
    gbdt = GBDTRegressor(
        learning_rate=learning_rate, 
        max_depth=max_tree_depth, 
        max_tree_nums=max_tree_nums, 
        loss = SquaresError()
    )
    gbdt.build_gbdt(dataset, pso.get_gbest().get_pbest())
    
    # #Predict the results for test dataset
    # predict = gbdt.predict(dataset.get_test_data())
    # #print the accuracy

    # print (((predict['predict_value'] - predict['label']) ** 2).mean() ** .5)
    out=[]
    for _ in range(10):
        test = dataset.get_test_data()
        test = test.sample(frac=1).reset_index(drop=True)
        predict = gbdt.predict(test)
        out.append(((predict['predict_value'] - predict['label']) ** 2).mean() ** .5)
    
    print("RMSE: %0.2f (+/- %0.2f)" % (np.mean(out), np.std(out) * 2))
    print(f"time taken is {end - start}")

    y = pso.get_gbest_records()
    x = np.arange(iterations)

    base = 'pso-search'
    if is_bin:
        name = base + '-' + name + '-bins' + '-p' + str(size_population) + '-i' + str(iterations) 
    else:
        name = base + '-' + name + '-p' + str(size_population) + '-i' + str(iterations) 
    print(name)
    print(y)

    pickle.dump((pso.get_gbest().get_pbest()), open('train_models/' + name + '.pkl', "wb"))
    plot_pso(x, y, 'images/' + name + '.svg')


def pso_run_pre(df, pretrain_file, target_name, pretrain_type = 'xgb'):
    feature_names = df.columns.values
    df = df.sample(frac=1).reset_index(drop=True)
    train = df[:int(df.shape[0] * 0.8)]
    test  = df[int(df.shape[0] * 0.8):]

    if pretrain_type == 'xgb':
        model = PreprocessXgbModel(pretrain_file)
        internal_splits = model.get_internal_splits()
    else:
        model = PreprocessSklModel(pretrain_file)
        internal_splits = model.get_internal_splits(feature_names)

    dataset = DataSet(train, test, target_name, standardize=False)
    print("encode start...")
    dataset.encode_table(pretrained=True, pretrain_arrays=internal_splits)
    print("encode done...")
    lookup_tables = dataset.get_lookup_table()

    pretrain_arrays = [dataset.get_index(val) for val in internal_splits]
    iterations =20
    size_population = 50
    max_tree_nums = 6
    learning_rate = 1
    max_tree_depth = 5

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
        model_type='regression', 
        beta=0.45, 
        alfa=0.45
    )
    pso.run() # runs the PSO algorithm
    print('gbest: %s | cost: %f\n' % (pso.get_gbest().get_pbest(), pso.get_gbest().get_cost_pbest()))

    end = time.time()

    #Build the gbdt with gbest solution
    gbdt = GBDTRegressor(
        learning_rate=learning_rate, 
        max_depth=max_tree_depth, 
        max_tree_nums=max_tree_nums, 
        loss = SquaresError()
    )
    gbdt.build_gbdt(dataset, pso.get_gbest().get_pbest())
    
    out = []
    for _ in range(10):
        test = dataset.get_test_data()
        test = test.sample(frac=1).reset_index(drop=True)
        predict = gbdt.predict(test)
        out.append(((predict['predict_value'] - predict['label']) ** 2).mean() ** .5)
    
    print("RMSE: %0.2f (+/- %0.2f)" % (np.mean(out), np.std(out) * 2))

    # #Predict the results for test dataset
    # predict = gbdt.predict(dataset.get_test_data())
    # #print the accuracy
    # print (((predict['predict_value'] - predict['label']) ** 2).mean() ** .5)
    print(f"time taken is {end - start}")

    y = pso.get_gbest_records()
    x = np.arange(iterations)

    base = 'pso-pre'
    name = base + '-' + re.findall(r"/(.*).p",pretrain_file)[0] + '-p' + str(size_population) + '-i' + str(iterations) 
    print(name)
    print(y)

    pickle.dump((pso.get_gbest().get_pbest()), open('train_models/' + name + '.pkl', "wb"))
    plot_pso(x, y, 'images/' + name + '.svg')






if __name__ == "__main__":
    pso_run('quality_wine')

    # pretrain_file = "pretrain_models/kc_house.pkl"
    # pretrain_file = "pretrain_models/insurance.pkl"

    # pretrain_file = "pretrain_models/red_wine.pkl"
    # pso_run_pre(df, pretrain_file, target_name, pretrain_type='xgb')