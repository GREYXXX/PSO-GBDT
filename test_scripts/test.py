# @Author XI RAO
# CITS4001 Research Project

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
# target_name = 'label'

# df = pd.read_csv('data/BankNote.csv')
# target_name = 'class'

# df = pd.read_csv('data/classification.csv')

df = pd.read_csv('data/wine.csv')
df['quality'] = df['quality'].apply(lambda x : 0 if x == "bad" else 1)
target_name = 'quality'

# df = pd.read_csv("/Users/xirao/data/training_30.csv")
# df['Label'] = df['Label'].apply(lambda x : 1 if x == 's' else 0)
# target_name = 'label'

# df = pd.read_csv("/Users/xirao/data/covat_0.3.csv")
# df = df.drop('Unnamed: 0', axis = 1)
# df['54'] = df['54'].apply(lambda x : 1 if x > 1.0 else 0)
# target_name = '54'

# df = pd.read_csv("/Users/xirao/data/higgs_0.005.csv")
# df = df.drop('Unnamed: 0', axis = 1)
# df['28'] = df['28'].apply(lambda x : int(x))
# target_name = '28'


df = df.sample(frac=1).reset_index(drop=True)
train = df[:int(df.shape[0] * 0.8)]
test  = df[int(df.shape[0] * 0.8):]

is_bin = False
dataset = DataSet(train, test, target_name, standardize=False, is_bin=is_bin)
print("encode start...")
dataset.encode_table()
print("encode done...")
lookup_tables = dataset.get_lookup_table()


def sklearn_train():
    # df = pd.read_csv("/Users/xirao/data/higgs_0.005.csv")
    # df = df.drop('Unnamed: 0', axis = 1)
    # X = df.drop('28', axis=1)
    # y = df['28']

    # df = pd.read_csv('data/BankNote.csv')
    # X = df.drop('class', axis=1)
    # y = df['class']

    # df = pd.read_csv("/Users/xirao/data/covat_0.3.csv")
    # df = df.drop('Unnamed: 0', axis = 1)
    # X = df.drop('54', axis=1)
    # y = df['54']

    df = pd.read_csv('data/wine.csv')
    X = df.drop('quality', axis=1)
    y = df['quality']

    # X = df.drop('success', axis=1)
    # y = df['success']
    # lr_list = [0.01, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    # lr_list = [0.1, 0.25, 0.4, 1]
    # lr_list = [1]

    learning_rate = 1
    gb  = []
    xb = []
    cat = [] 
    dt  = []

    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        gb_clf = GradientBoostingClassifier(n_estimators=5, learning_rate=learning_rate, max_depth=4, random_state=0)
        gb_clf.fit(X_train, y_train)
        gb.append(gb_clf.score(X_test, y_test))

        # print("Learning rate: ", learning_rate)
        # print("SKR Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
        # print("SKR Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))

        xgbd = xgb.XGBClassifier(n_estimators=5, learning_rate=learning_rate, max_depth=4)
        xgbd.fit(X_train, y_train)
        xb.append(xgbd.score(X_test, y_test))

        # print("Learning rate: ", learning_rate)
        # print("XGB Accuracy score (training): {0:.3f}".format(xgbd.score(X_train, y_train)))
        # print("XGB Accuracy score (validation): {0:.3f}".format(xgbd.score(X_test, y_test)))

        clf = CatBoostClassifier(
            iterations=5, 
            depth = 4,
            learning_rate=1, 
        )
        clf.fit(X_train, y_train)
        cat.append(clf.score(X_test, y_test))

        # print("Learning rate: ", learning_rate)
        # print("Cat Accuracy score (training): {0:.3f}".format(clf.score(X_train, y_train)))
        # print("Cat Accuracy score (validation): {0:.3f}".format(clf.score(X_test, y_test)))


    # pickle.dump(xgbd, open("pretrain_models/wine.pkl", "wb"))
    # pickle.dump(gb_clf, open("pretrain_models/wine_sk.pkl", "wb"))

    # df = xgbd.get_booster().trees_to_dataframe()
    # print(df)

    # dump_list=xgbd.get_booster().get_dump()
    # tree1 = dump_list[0]
    # print(tree1)
    # for i in dump_list:
    #     print(i)

        decisionTreeModel = DecisionTreeClassifier(criterion= 'entropy',
                                            max_depth = 7, 
                                            splitter='best', 
                                            random_state=10)

        decisionTreeModel.fit(X_train,y_train)
        dt.append(decisionTreeModel.score(X_test, y_test))
        # print(' Train Score is   : ' ,decisionTreeModel.score(X_train, y_train))
        # print(' Test Score is    : ' ,decisionTreeModel.score(X_test, y_test))
  
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(gb), np.std(gb) * 2))
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(xb), np.std(xb) * 2))
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(cat), np.std(cat) * 2))
    print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(dt), np.std(dt) * 2))

def run():
    a = []
    gbdt = GBDTBinaryClassifier(learning_rate= 0.4, max_depth= 4, max_tree_nums= 4, loss = BinomialDeviance())
    # gbdt = GBDTRegressor(0.1, 4, 2, SquaresError())
    gbdt.build_gbdt(dataset, [])
    array = gbdt.get_gbdt_array()
    print(array)
    print(f"residuals are : {gbdt.get_residuals()}")
    # print(dataset.get_train_data())

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

    base = 'pso-search'
    if is_bin:
        name = base + '-' + name + '-bins' + '-p' + str(size_population) + '-i' + str(iterations) 
    else:
        name = base + '-' + name + '-p' + str(size_population) + '-i' + str(iterations) 
    print(name)

    pickle.dump((pso.get_gbest().get_pbest()), open('train_models/' + name + '.pkl', "wb"))
    plot_pso(x, y, 'images/' + name + '.svg')


# sklearn_train()
pso_run('wine')
# run()
# test_predict()
# test_xgb_predict()

