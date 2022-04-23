import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.tree import export_graphviz
import time
from data import DataSet
from GBDT import GBDTBinaryClassifier, GBDTRegressor
from pso import PSO
import pickle
from loss import BinomialDeviance, SquaresError
from ProcessModel import PreprocessXgbModel, PreprocessSklModel
import matplotlib.pyplot as plt
import re

import warnings
warnings.filterwarnings("ignore")


def plot_pso(x, y, name):
    plt.grid(which='major', axis='y', color='darkgray')
    plt.grid(which='major', axis='x', color='darkgray')
    plt.plot(x,y,color='fuchsia', marker='P', linestyle='dashed')
    plt.xlabel('Number of iteration',size = 15)
    plt.ylabel('Fitness value',size = 15)
    plt.title('PSO with pretrain')
    # plt.show() 
    plt.savefig(name)

def pso_run(df, pretrain_file, target_name, pretrain_type = 'xgb'):
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
    iterations =10
    size_population = 30
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
        model_type='binary_cf', 
        beta=0.45, 
        alfa=0.45
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

    y = pso.get_gbest_records()
    x = np.arange(iterations)
    # name = 'pso-pre-convat_0.3-p50'

    base = 'pso-pre'
    name = base + '-' + re.findall(r"/(.*).p",pretrain_file)[0] + '-p' + str(size_population) + '-i' + str(iterations) 
    print(name)

    pickle.dump((pso.get_gbest().get_pbest()), open('train_models/' + name + '.pkl', "wb"))
    plot_pso(x, y, 'images/' + name + '.svg')





if __name__ == "__main__":

    # df = pd.read_csv("/Users/xirao/data/covat_0.3.csv")
    # df = df.drop('Unnamed: 0', axis = 1)
    # df['54'] = df['54'].apply(lambda x : 1 if x > 1.0 else 0)
    # pretrain_file = 'pretrain_models/covat_0.3.pkl'
    # target_name = '54'

    # df = pd.read_csv("data/wine.csv")
    # df['quality'] = df['quality'].apply(lambda x : 0 if x == "bad" else 1)
    # target_name = 'quality'
    # pretrain_file = 'pretrain_models/wine.pkl'
    
    # df = pd.read_csv('data/BankNote.csv')
    # pretrain_file = 'pretrain_models/BankNotes.pkl'
    # target_name = 'class'

    df = pd.read_csv("/Users/xirao/data/higgs_0.005.csv")
    df = df.drop('Unnamed: 0', axis = 1)
    df['28'] = df['28'].apply(lambda x : int(x))
    target_name = '28'
    pretrain_file = "pretrain_models/higgs.pkl"

    # pretrain_file = 'pretrain_models/wine_sk.pkl'
    # target_name = 'quality'

    pso_run(df, pretrain_file, target_name, pretrain_type='xgb')