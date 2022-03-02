from data import DataSet
from loss import SquaresError, BinomialDeviance, MultinomialDeviance
from GBDT import GBDTRegressor, GBDTBinaryClassifier, GBDTMultiClassifier
import pandas as pd
from data import DataSet
import random
from pso import PSO
import xgboost as xgb
from xgboost import plot_tree
import os
import pickle
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree as sktree
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# df = pd.read_csv('data/credit.data.csv')
# train = df[:int(df.shape[0] * 0.8)]
# test  = df[int(df.shape[0] * 0.8):]
# target_name = 'label'

# df = pd.read_csv('data/BankNote.csv')
# df = df.sample(frac=1).reset_index(drop=True)
# train = df[:int(df.shape[0] * 0.8)]
# test  = df[int(df.shape[0] * 0.8):]
# target_name = 'class'

# df = pd.read_csv('data/classification.csv')

# df = pd.read_csv('data/wine.csv')
# df['quality'] = df['quality'].apply(lambda x : 0 if x == "bad" else 1)

# df = pd.read_csv('data/heart_disease.csv')
# x = df.drop('HeartDiseaseorAttack', axis=1)
# y = df['HeartDiseaseorAttack']
# x['HeartDiseaseorAttack'] = y
# df = x

# df = pd.read_csv('data/Swarm_Behaviour.csv')
# target_name = 'success'

# df = pd.read_csv("/Users/xirao/data/training_30.csv")
# df['Label'] = df['Label'].apply(lambda x : 1 if x == 's' else 0)
# train = df[:int(df.shape[0] * 0.8)]
# test  = df[int(df.shape[0] * 0.8):]
# target_name = 'label'

# df = pd.read_csv("/Users/xirao/data/covat_0.3.csv")
# df['54'] = df['54'].apply(lambda x : 1 if x > 1.0 else 0)
# train = df[:int(df.shape[0] * 0.8)]
# test  = df[int(df.shape[0] * 0.8):]
# target_name = '54'

df = pd.read_csv("/Users/xirao/data/covat_0.3.csv")
df = df.drop('Unnamed: 0', axis = 1)
df['54'] = df['54'].apply(lambda x : 1 if x > 1.0 else 0)
train = df[:int(df.shape[0] * 0.8)]
test  = df[int(df.shape[0] * 0.8):]
target_name = '54'

dataset = DataSet(train, test, target_name, standardize=False)
print("encode start...")
dataset.encode_table()
print("encode done...")
lookup_tables = dataset.get_lookup_table()

def sklearn_train():
    # X, y = pickle.load(open('/Users/xirao/data/higgs.plk', "rb"))
    # df = pd.read_csv("/Users/xirao/data/higgs_0.005.csv")
    df = pd.read_csv("/Users/xirao/data/covat_0.3.csv")
    # X = df.drop('success', axis=1)
    # y = df['success']
    X = df.drop('54', axis=1)
    y = df['54']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print(X_train)
    # lr_list = [0.01, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    # lr_list = [0.1, 0.25, 0.4, 1]
    lr_list = [1]

    for learning_rate in lr_list:
        # gb_clf = GradientBoostingClassifier(n_estimators=6, learning_rate=learning_rate, max_depth=5, random_state=0)
        # gb_clf.fit(X_train, y_train)

        # print("Learning rate: ", learning_rate)
        # print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
        # print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))

        xgbd = xgb.XGBClassifier(n_estimators=6, learning_rate=learning_rate, max_depth=6)
        xgbd.fit(X_train, y_train)

        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(xgbd.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(xgbd.score(X_test, y_test)))

    # df = xgbd.get_booster().trees_to_dataframe()
    # print(df)

    dump_list=xgbd.get_booster().get_dump()
    tree1 = dump_list[0]
    print(tree1)
    # for i in dump_list:
    #     print(i)

    # decisionTreeModel = DecisionTreeClassifier(criterion= 'entropy',
    #                                        max_depth = None, 
    #                                        splitter='best', 
    #                                        random_state=10)

    # decisionTreeModel.fit(X_train,y_train)
    # print(' Train Score is   : ' ,decisionTreeModel.score(X_train, y_train))
    # print(' Test Score is    : ' ,decisionTreeModel.score(X_test, y_test))

def run():
    a = []
    gbdt = GBDTBinaryClassifier(learning_rate= 0.4, max_depth= 4, max_tree_nums= 4, loss = BinomialDeviance())
    # gbdt = GBDTRegressor(0.1, 4, 2, SquaresError())
    gbdt.build_gbdt(dataset, [])
    array = gbdt.get_gbdt_array()
    print(array)
    print(f"residuals are : {gbdt.get_residuals()}")

    random.shuffle(array)
    print(gbdt.get_gbdt_array_all())
    predict = gbdt.predict(dataset.get_test_data())
    print(sum(predict['label'] == predict['predict_label']) / len(predict))

    print("\n")
    print("shuffled: {}".format(array))
    gbdt_ = GBDTBinaryClassifier(learning_rate= 0.4, max_depth= 4, max_tree_nums= 1, loss = BinomialDeviance())
    gbdt_.build_gbdt(dataset, array)
    print(gbdt_.get_gbdt_array_all())
    print(f"residuals shuffled are : {gbdt.get_residuals()}")

    predict = gbdt_.predict(dataset.get_test_data())
    print(sum(predict['label'] == predict['predict_label']) / len(predict))

def test_predict():
    models = []
    accs   = []
    print("gbdt build starting")
    b_start = time.time()
    for _ in range(50):
        gbdt = GBDTBinaryClassifier(learning_rate= 0.4, max_depth= 4, max_tree_nums= 6, loss = BinomialDeviance())
        # gbdt = GBDTRegressor(0.1, 4, 2, SquaresError())
        gbdt.build_gbdt(dataset, [])
        models.append(gbdt)
    print("gbdt build done and predcit starting")
    b_end = time.time()
    print(f"building time end with {b_end - b_start}")

    count = 0
    start1 = time.time()
    for model in models:
        start = time.time()
        predict = model.predict(dataset.get_test_data())
        end = time.time()
        accs.append(sum(predict['label'] == predict['predict_label']) / len(predict))
        print(f"model {count} done with time {end - start}")
        count += 1
    end1 = time.time()
    
    print(f"gbdt predict done with time {end1 - start1}")

def test_xgb_predict():
    models = []
    accs   = []
    df = pd.read_csv("/Users/xirao/data/covat_0.3.csv")
    X = df.drop('54', axis=1)
    y = df['54']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("gbdt build starting")
    b_start = time.time()
    for _ in range(50):
        # xgbd = xgb.XGBClassifier(n_estimators=6, learning_rate=0.4, max_depth=4)
        # xgbd.fit(X_train, y_train)
        # models.append(xgbd)

        gb_clf = GradientBoostingClassifier(n_estimators=6, learning_rate=0.4, max_depth=4, random_state=0)
        gb_clf.fit(X_train, y_train)
        models.append(gb_clf)
    b_end = time.time()
    print(f"building time end with {b_end - b_start}")

    print("gbdt build done and predcit starting")

    count = 0
    start = time.time()
    for model in models:
        start1 = time.time()
        predict = model.predict(X_test)
        end1 = time.time()
        # evaluate predictions
        accuracy = accuracy_score(y_test, predict)
        accs.append(accuracy)
        print(f"model {count} done with time {end1 - start1}")
        count += 1
    end = time.time()
    
    print(f"gbdt predict done with time {end - start}")
    print(accs)


def pso_run():
    """dataset, iterations, size_population, max_tree_nums=5, learning_rate = 0.3, max_tree_depth = 4, model_type='regression', beta=1, alfa=1"""

    iterations=10
    size_population=30
    max_tree_nums=5
    learning_rate = 0.4
    max_tree_depth = 5

    start = time.time()
    pso = PSO(dataset, iterations=iterations, size_population=size_population,  max_tree_nums=max_tree_nums, learning_rate = learning_rate, 
                                max_tree_depth = max_tree_depth, model_type='binary_cf', beta=0.3, alfa=0.3)
    pso.run() # runs the PSO algorithm
    print('gbest: %s | cost: %f\n' % (pso.get_gbest().get_pbest(), pso.get_gbest().get_cost_pbest()))

    end = time.time()

    #Build the gbdt with gbest solution
    gbdt = GBDTBinaryClassifier(learning_rate=learning_rate, max_depth=max_tree_depth, max_tree_nums=max_tree_nums, loss = BinomialDeviance())
    gbdt.build_gbdt(dataset, pso.get_gbest().get_pbest())
    #Predict the results for test dataset
    predict = gbdt.predict(dataset.get_test_data())
    #print the accuracy
    print(sum(predict['label'] == predict['predict_label']) / len(predict))
    print(f"time taken is {end - start}")


# sklearn_train()
pso_run()
# run()
# test_predict()
# test_xgb_predict()

