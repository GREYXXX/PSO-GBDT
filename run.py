
from typing import Tuple
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
from ProcessModel import PreprocessXgbModel, PreprocessSklModel
import os
import pickle
import time
from typing import Tuple
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import argparse

import warnings
warnings.filterwarnings("ignore")

def get_feature_and_targets(
    filename : str
    ) -> Tuple[pd.DataFrame, pd.Series]:

    zip_file = 'data.zip'
    base = 'data/'
    EXIST_FILE_NAMES = {
        'BankNote.csv' : 'class', 
        'insurance.csv' : 'charges', 
        'wine.csv' : 'quality', 
        'winequality-red.csv' : 'quality',
        'higgs_0.005.csv' : '28',
        'covat_0.3.csv' : '54',
        'kc_house_data.csv' : 'price'
        }

    if filename in EXIST_FILE_NAMES.keys():
        target_name = EXIST_FILE_NAMES[filename]
        with zipfile.ZipFile(zip_file) as zf:
            df = pd.read_csv(zf.open(base + filename))
            X = df.drop(target_name, axis=1)
            y = df[target_name]
    
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=os.getenv('DATASET_PATH', 'BankNote.csv'))
    parser.add_argument('--model_type', type=str, default=os.getenv('MODEL_TYPE', 'binary_cf'))
    parser.add_argument('--model_dir', type=str, default=os.getenv('MODEL_DIR', 'models/'))
    parser.add_argument('--size_population', type=int, default=os.getenv('PSO_POPULATION_SIZE', 50))
    parser.add_argument('--iterations', type=int, default=os.getenv('PSO_ITERATIONS', 20))
    parser.add_argument('--lr', type=float, default=os.getenv('LEARNING_RATE', 1))
    parser.add_argument('--max_tree_nums', type=int, default=os.getenv('MAX_TREE_NUM', 6))
    parser.add_argument('--max_tree_depth', type=int, default=os.getenv('MAX_TREE_DEPTH', 5))
    parser.add_argument('--num_bins', type=int, default=os.getenv('NUM_OF_BINS', -1))
    parser.add_argument('--pretrain_file', type=str, default=os.getenv('PRETRAIN_FILE', ''))
    parser.add_argument('--pretrain_type', type=str, default=os.getenv('PRETRAIN_TYPE', 'skl'))

    # parser.add_argument('--alpha', type=float, default=os.getenv('ALPHA_FOR_PSO', 0.45))
    # parser.add_argument('--beta', type=float, default=os.getenv('BETA_FOR_PSO', 0.45))

    base_pretrain_path = 'pretrain_models/'
    args, _ = parser.parse_known_args()
    print(os.path.join(base_pretrain_path, args.pretrain_file))
    print(f"dataset path is {args.dataset_path}")
    X, y = get_feature_and_targets(args.dataset_path)

    # Init pso trainer
    pso = PSO(
        iterations = args.iterations, 
        size_population = args.size_population,  
        max_tree_nums = args.max_tree_nums, 
        learning_rate = args.lr, 
        max_tree_depth = args.max_tree_depth, 
        model_type=args.model_type
    )

    if args.pretrain_file != '':
        if args.pretrain_type == 'xgb':
            model = PreprocessXgbModel(os.path.join(base_pretrain_path, args.pretrain_file))
            internal_splits = model.get_internal_splits()
        elif args.pretrain_type == 'skl':
            model = PreprocessSklModel(os.path.join(base_pretrain_path, args.pretrain_file))
            internal_splits = model.get_internal_splits(X.columns.values)
        else:
            raise ValueError('This implementation only accept pretrain type for either xgb or skr')

        pso.run(X, y, internal_splits, use_pretrain=True)
    else:
        pso.run(X, y)
    
    print('gbest: %s | cost: %f\n' % (pso.get_gbest().get_pbest(), pso.get_gbest().get_cost_pbest()))






