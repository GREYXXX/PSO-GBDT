from cProfile import label
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
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier



# df = pd.read_csv('data/credit.data.csv')
# df = pd.read_csv('data/BankNote.csv')
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

df = pd.read_csv("/Users/xirao/data/covat_0.3.csv")
df['54'] = df['54'].apply(lambda x : 1 if x > 1.0 else 0)
train = df[:int(df.shape[0] * 0.8)]
test  = df[int(df.shape[0] * 0.8):]
target_name = '54'


dataset = DataSet(train, test, target_name,standardize=False)
print("encode start...")
dataset.encode_table()
print("encode done...")
lookup_tables = dataset.get_lookup_table()

def sklearn_train():
    print(df)
    # X = df.drop('success', axis=1)
    # y = df['success']
    X = df.drop('Label', axis=1)
    y = df['Label']
    # X = df.drop('quality', axis=1)
    # y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(type(X_train))
    # lr_list = [0.01, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    lr_list = [0.1, 0.25, 0.4]

    for learning_rate in lr_list:
        gb_clf = GradientBoostingClassifier(n_estimators=6, learning_rate=learning_rate, max_depth=6, random_state=0)
        gb_clf.fit(X_train, y_train)

        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))

    decisionTreeModel = DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = None, 
                                           splitter='best', 
                                           random_state=10)

    decisionTreeModel.fit(X_train,y_train)
    print(' Train Score is   : ' ,decisionTreeModel.score(X_train, y_train))
    print(' Test Score is    : ' ,decisionTreeModel.score(X_test, y_test))

def run():
    gbdt = GBDTBinaryClassifier(learning_rate= 0.4, max_depth= 4, max_tree_nums= 1, loss = BinomialDeviance())
    # gbdt = GBDTRegressor(0.1, 4, 2, SquaresError())
    gbdt.build_gbdt(dataset, [])
    array = gbdt.get_gbdt_array()
    print(array)
    random.shuffle(array)
    # print([lookup_tables[i] for i in gbdt.get_gbdt_array()()])
    # print(gbdt.get_trees_loss())
    print(gbdt.get_gbdt_array_all())
    predict = gbdt.predict(dataset.get_test_data())
    print(sum(predict['label'] == predict['predict_label']) / len(predict))

    print("\n")
    print("shuffled: {}".format(array))
    gbdt_ = GBDTBinaryClassifier(learning_rate= 0.4, max_depth= 4, max_tree_nums= 1, loss = BinomialDeviance())
    gbdt_.build_gbdt(dataset, array)
    print(gbdt_.get_gbdt_array_all())

    predict = gbdt_.predict(dataset.get_test_data())
    #print(((predict['predict_value'] - predict['label']) ** 2).mean() ** .5)
    print(sum(predict['label'] == predict['predict_label']) / len(predict))
    

def pso_run():
    """dataset, iterations, size_population, max_tree_nums=5, learning_rate = 0.3, max_tree_depth = 4, model_type='regression', beta=1, alfa=1"""

    iterations=10
    size_population=20
    max_tree_nums=6
    learning_rate = 0.4
    max_tree_depth = 5

    pso = PSO(dataset, iterations=iterations, size_population=size_population,  max_tree_nums=max_tree_nums, learning_rate = learning_rate, 
                                max_tree_depth = max_tree_depth, model_type='binary_cf', beta=0.5, alfa=0.5)
    pso.run() # runs the PSO algorithm
    print('gbest: %s | cost: %f\n' % (pso.get_gbest().get_pbest(), pso.get_gbest().get_cost_pbest()))

    #Build the gbdt with gbest solution
    gbdt = GBDTBinaryClassifier(learning_rate=learning_rate, max_depth=max_tree_depth, max_tree_nums=max_tree_nums, loss = BinomialDeviance())
    gbdt.build_gbdt(dataset, pso.get_gbest().get_pbest())
    #Predict the results for test dataset
    predict = gbdt.predict(dataset.get_test_data())
    #print the accuracy
    print(sum(predict['label'] == predict['predict_label']) / len(predict))


# sklearn_train()
pso_run()
# run()

