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
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier

# df = pd.read_csv('data/credit.data.csv')
df = pd.read_csv('data/BankNote.csv')
# df = pd.read_csv('data/classification.csv')

# df = pd.read_csv('data/heart_disease.csv')
# x = df.drop('HeartDiseaseorAttack', axis=1)
# y = df['HeartDiseaseorAttack']
# x['HeartDiseaseorAttack'] = y
# df = x

# df = pd.read_csv('data/Swarm_Behaviour.csv')


dataset = DataSet(df)
print("encode start...")
dataset._encodeTable()
print("encode done...")
lookup_tables = dataset.getLookupTable()

def sklearn_train():
    X = df.drop('Swarm_Behaviour', axis=1)
    y = df['Swarm_Behaviour']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

    for learning_rate in lr_list:
        gb_clf = GradientBoostingClassifier(n_estimators=4, learning_rate=learning_rate, max_depth=5, random_state=0)
        gb_clf.fit(X_train, y_train)

        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))

def run():
    gbdt = GBDTBinaryClassifier(learning_rate= 0.5, max_depth= 4, max_tree_nums= 4, loss = BinomialDeviance())
    # gbdt = GBDTRegressor(0.1, 4, 2, SquaresError())
    gbdt._build_gbdt(dataset, [])
    array = gbdt.getGBDTArray()
    random.shuffle(array)
    print(gbdt.getGBDTArray())
    print([lookup_tables[i] for i in gbdt.getGBDTArray()])
    print(gbdt.getGBDTArrayAll())

    print("\n")
    print("shuffled: {}".format(array))
    gbdt._build_gbdt(dataset, array)
    print(gbdt.getGBDTArray())
    print(gbdt.getGBDTArrayAll())

    predict = gbdt.predict(df).iloc[:,15:]
    predict = gbdt.predict(dataset.getTestData())
    #print(((predict['predict_value'] - predict['label']) ** 2).mean() ** .5)
    print(sum(predict['label'] == predict['predict_label']) / len(predict))
    

def pso_run():
    """dataset, iterations, size_population, max_tree_nums=5, learning_rate = 0.3, max_tree_depth = 4, model_type='regression', beta=1, alfa=1"""

    iterations=10 
    size_population=20
    max_tree_nums=5
    learning_rate = 0.3
    max_tree_depth = 4

    pso = PSO(dataset, iterations=iterations, size_population=size_population,  max_tree_nums=max_tree_nums, learning_rate = learning_rate, 
                                max_tree_depth = max_tree_depth, model_type='binary_cf', beta=0.8, alfa=0.8)
    pso.run() # runs the PSO algorithm
    print('gbest: %s | cost: %f\n' % (pso.getGBest().getPBest(), pso.getGBest().getCostPBest()))

    #Build the gbdt with gbest solution
    gbdt = GBDTBinaryClassifier(learning_rate=learning_rate, max_depth=max_tree_depth, max_tree_nums=max_tree_nums, loss = BinomialDeviance())
    gbdt._build_gbdt(dataset, pso.getGBest().getPBest())
    #Predict the results for test dataset
    predict = gbdt.predict(dataset.getTestData())
    print(sum(predict['label'] == predict['predict_label']) / len(predict))



# sklearn_train()
pso_run()
# run()