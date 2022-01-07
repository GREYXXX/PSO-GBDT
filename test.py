from numpy import array
from data import DataSet
from loss import SquaresError, BinomialDeviance, MultinomialDeviance
from GBDT import GBDTRegressor, GBDTBinaryClassifier, GBDTMultiClassifier
import pandas as pd
from data import DataSet
import random
from pso import PSO

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree as sktree

df = pd.read_csv('data/credit.data.csv')
dataset = DataSet(df)
dataset._encodeTable()

def run():
    gbdt = GBDTBinaryClassifier(0.1, 4, 2, BinomialDeviance())
    # gbdt = GBDTRegressor(0.1, 4, 2, SquaresError())
    gbdt._build_gbdt(dataset, [])
    array = gbdt.getGBDTArray()
    random.shuffle(array)
    print(gbdt.getGBDTArray())
    print(gbdt.getGBDTArrayAll())

    print("\n")
    print("shuffled: {}".format(array))
    gbdt._build_gbdt(dataset, array)
    print(gbdt.getGBDTArray())
    print(gbdt.getGBDTArrayAll())

    predict = gbdt.predict(df).iloc[:,15:]
    #print(((predict['predict_value'] - predict['label']) ** 2).mean() ** .5)
    print(sum(predict['label'] == predict['predict_label']) / len(predict))

def pso_run():
    """dataset, iterations, size_population, max_tree_nums=5, model_type='regression', beta=1, alfa=1"""
    pso = PSO(dataset, 10, 50,  max_tree_nums=5, model_type='binary_cf', beta=0.5, alfa=0.5)
    pso.run() # runs the PSO algorithm
    print('gbest: %s | cost: %f\n' % (pso.getGBest().getPBest(), pso.getGBest().getCostPBest()))


pso_run()
# run()