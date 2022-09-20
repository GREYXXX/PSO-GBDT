
from typing import List, Tuple
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
import json

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
            if filename == 'wine.csv':
                df[target_name] = df[target_name].apply(lambda x : 0 if x == "bad" else 1)
            elif filename == 'higgs_0.005.csv':
                df[target_name] = df[target_name].apply(lambda x : int(x))
            elif filename == 'covat_0.3.csv':
                df[target_name] = df[target_name].apply(lambda x : 1 if x > 1.0 else 0)
            
            X = df.drop(target_name, axis=1)
            y = df[target_name]
    
    return X, y

def inference(
    args, 
    dataset : DataSet,
    gbest_arrary : List[int],
    )-> float:

    if args.model_type == 'regression':
        gbdt = GBDTRegressor(
            args.lr, 
            args.max_tree_depth, 
            args.max_tree_nums, 
            SquaresError()
        )

    elif args.model_type == 'binary_cf':
        gbdt = GBDTBinaryClassifier(
            args.lr, 
            args.max_tree_depth, 
            args.max_tree_nums, 
            BinomialDeviance()
        )
            
    elif args.model_type == 'multi_cf':
        gbdt = GBDTMultiClassifier(
            args.lr, 
            args.max_tree_depth, 
            args.max_tree_nums, 
            MultinomialDeviance()
        )
    else:
        raise ValueError("Invalid model type. Requires a valid model type: regression, binary_cf or multi_cf")

    gbdt.build_gbdt(dataset, gbest_arrary)

    #Predict the results for test dataset
    test_data = dataset.get_test_data()
    predict = gbdt.predict(test_data)

    #print the accuracy
    test_acc = sum(predict['label'] == predict['predict_label']) / len(predict)
    print(test_acc)

    return test_acc

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
    parser.add_argument('--pretrain_type', type=str, default=os.getenv('PRETRAIN_TYPE', ''))
    parser.add_argument('--use_validation', type=bool, default=os.getenv('IF_USE_VALIDATION', True))
    parser.add_argument('--direct_pretrain', type=bool, default=os.getenv('IF_USE_DIRECT_PRATRAIN_RESULT', False))

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

    if args.pretrain_file != '' and not args.direct_pretrain:
        file = os.path.join(base_pretrain_path, args.pretrain_file)
        pretrain_model = pickle.load(open(file, "rb"))
        if args.pretrain_type == 'xgb':
            model = PreprocessXgbModel(pretrain_model)
            internal_splits = model.get_internal_splits()
        elif args.pretrain_type == 'skl':
            model = PreprocessSklModel(pretrain_model)
            internal_splits = model.get_internal_splits(X.columns.values)
        else:
            raise ValueError('This implementation only accept pretrain type for either xgb or skr')

        pso.run(
            X, 
            y, 
            internal_splits=internal_splits, 
            use_pretrain=True, 
            use_validation=args.use_validation,
        )
        
    elif args.pretrain_file == '' and args.direct_pretrain:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if args.pretrain_type == 'xgb':
            print('pretraining with XGBoost start ...')
            xgbd = xgb.XGBClassifier(
                n_estimators=args.max_tree_nums, 
                learning_rate=args.lr, 
                max_depth=args.max_tree_depth - 1
            )
            xgbd.fit(X_train, y_train)
            model = PreprocessXgbModel(model=xgbd)
            internal_splits = model.get_internal_splits()

        elif args.pretrain_type == 'skr':
            print('pretraining with Sklearn GBM start ...')
            skgb = GradientBoostingClassifier(
                n_estimators=args.max_tree_nums, 
                learning_rate=args.lr, 
                max_depth=args.max_tree_depth - 1
            )
            skgb.fit(X_train, y_train)
            model = PreprocessXgbModel(model=skgb)
            internal_splits = model.get_internal_splits()
        
        print('pretraining done ...')
        pso.run(
            X, 
            y, 
            internal_splits=internal_splits, 
            use_pretrain=True, 
            use_validation=args.use_validation,
        )

    else:
        pso.run(
            X, 
            y,
            use_validation=args.use_validation,
        )

    gbest_array = pso.get_gbest().get_pbest()
    gbest_cost = pso.get_gbest().get_cost_pbest()
    print('gbest: %s | cost: %f\n' % (gbest_array, gbest_cost))

    test_acc = inference(
        args = args,
        dataset = pso.dataset,
        gbest_arrary = gbest_array,
    )
    
    result = {
        'dataset_name' : args.dataset_path,
        'model_type': args.model_type,
        'size_population' : args.size_population,
        'learning_rate' : args.lr,
        'max_tree_nums' : args.max_tree_nums,
        'max_tree_depth' : args.max_tree_depth,
        'training_acc (gbest_cost)' : gbest_cost,
        'num_of_bins' : args.num_bins,
        'iterations' : args.iterations,
        'pretrain_type': args.pretrain_type,
        'testing_acc' : test_acc,
        'gbest_sequence' : gbest_array
    }

    result_dir_path = 'result/'
    if not os.path.isdir(result_dir_path):
        os.mkdir(result_dir_path)

    if not args.pretrain_file:
        out_name = os.path.join(result_dir_path, args.dataset_path.split('.')[0] + "_result.json")
    else:
        out_name = os.path.join(result_dir_path, args.dataset_path.split('.')[0] + "_pretrain_result.json")

    with open(out_name, "w") as outfile:
        json.dump(result, outfile)

    # pickle.dump((pso.get_gbest().get_pbest()), open('train_models/' + args.dataset_path.split('.')[0] + '.pkl', "wb"))