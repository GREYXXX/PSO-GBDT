from Tree import Tree
import pandas as pd
import warnings
import time
import numpy as np
warnings.filterwarnings("ignore")


class BaseGBDT:

    def __init__(self, learning_rate, max_depth, max_tree_nums, loss):

        """
        param learning_rate: learning rate
        param max_depth: max depth of the decision tree
        param max_tree_nums: the number of the boosting trees
        param loss: the loss type (SquaresError, BinomialDeviance, MultinomialDeviance)
        """

        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_tree_nums = max_tree_nums
        self.loss = loss
        self.f_0 = {}
        self.trees = {}
        self.tree_loss = []
        self.residuals = []

    def build_gbdt(self, dataset, tree_array):
        data = dataset.get_train_data()
        self.f_0 = self.loss.initialize_f_0(data)
        cut_tree_array = []
        
        # If not at init stage, cut the GBDT array for multiple tree arrays
        if len(tree_array) != 0:
            cut_tree_array = [tree_array[i:i+self.max_depth] for i in range(0,len(tree_array),self.max_depth)]
        
        # If at the init stage, randomly init the nodes of GBDT with different feature at every level
        else:
            for _ in range(self.max_tree_nums):
                drops = []
                vals = []
                for _ in range(self.max_depth):
                    merge = dataset.get_random_elements(drops)
                    vals.append(dataset.get_index(merge))
                    drops.append(merge[0])
                cut_tree_array.append(vals)

        # Construct the GBDT
        for i in range(1, self.max_tree_nums + 1):
            self.loss.calculate_residual(data, i)
            target_name = 'res_' + str(i)
            tree = Tree(dataset, target_name, self.loss, cut_tree_array[i-1], max_depth = self.max_depth, tree_id = i)
            tree.build_tree(data)
            self.trees[i] = tree
            # self.loss.update_f_m(data, self.trees, i, self.learning_rate)
            self.residuals.append(self.loss.update_f_m(data, self.trees, i, self.learning_rate))
        

    def get_residuals(self):
        """return residuals"""
        return self.residuals

    def get_trees_loss(self):
        """return the loss of trees"""
        return self.tree_loss

    def get_gbdt_array(self):
        """
        Encode the GBDT by (feature, feature_val) --> key 
        """
        gbdt_array = [self.trees[i].get_tree_array() for i in range(1, len(self.trees) + 1)]
        return [i for item in gbdt_array for i in item]
    
    def  get_gbdt_array_all(self):
        """
        Encode the GBDT by (feature, feature_val) --> key, with calling get_tree_arrayAll()
        """
        gbdt_array = [self.trees[i].get_tree_array_all() for i in range(1, len(self.trees) + 1)]
        return [i for item in gbdt_array for i in item]
    
    def get_gbdt_node_array(self):
        """
        Encode the GBDT by node --> Class Node()
        """
        gbdt_nodes_array = [self.trees[i].get_tree_nodes() for i in range(1, len(self.trees) + 1)]
        return [i for item in gbdt_nodes_array for i in item]


class GBDTRegressor(BaseGBDT):

    def __init__(self, learning_rate, max_depth, max_tree_nums, loss):
        super().__init__(learning_rate, max_depth, max_tree_nums, loss)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.max_tree_nums + 1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            leafs = self.trees[iter].predict(data)
            data[f_m_name] = data[f_prev_name].values + (self.learning_rate * leafs)

        data['predict_value'] = data[f_m_name]
        return data


class GBDTBinaryClassifier(BaseGBDT):

    def __init__(self, learning_rate, max_depth, max_tree_nums, loss):
        super().__init__(learning_rate, max_depth, max_tree_nums, loss)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.max_tree_nums + 1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            leafs = self.trees[iter].predict(data)
            data[f_m_name] = data[f_prev_name].values + (self.learning_rate * leafs)

        #apply sigmoid function
        condlist   = [1 / (1 + np.exp(-data[f_m_name])) >= 0.5] 
        choicelist = [1]
        data['predict_label'] = np.select(condlist, choicelist, default = 0)

        return data


class GBDTMultiClassifier(BaseGBDT):

    def __init__(self, learning_rate, max_depth, max_tree_nums, loss):
        super().__init__(learning_rate, max_depth, max_tree_nums, loss)

    def build_gbdt(self, data):
        self.classes = data['label'].unique().astype(str)
        self.loss.init_classes(self.classes)
        
        # One-hot encoding for each class
        for class_name in self.classes:
            label_name = 'label_' + class_name
            data[label_name] = data['label'].apply(lambda x: 1 if str(x) == class_name else 0)
            
            # Init f_0(x)
            self.f_0[class_name] = self.loss.initialize_f_0(data, class_name)

        for i in range(1, self.max_tree_nums + 1):
            self.loss.calculate_residual(data, i)
            self.trees[iter] = {}

            for class_name in self.classes:
                target_name = 'res_' + class_name + '_' + str(iter)
                tree = Tree(data, target_name, self.loss, max_depth = 4, tree_id = i)
                tree.build_tree(data)
                self.trees[i][class_name] = tree
                self.loss.update_f_m(data, self.trees, i, class_name, self.learning_rate)


    def predict(self, data):
        for class_name in self.classes:
            f_0_name = 'f_' + class_name + '_0'
            data[f_0_name] = self.f_0[class_name]
            for iter in range(1, self.n_trees + 1):
                f_prev_name = 'f_' + class_name + '_' + str(iter - 1)
                f_m_name = 'f_' + class_name + '_' + str(iter)
                data[f_m_name] = \
                    data[f_prev_name] + \
                    self.learning_rate * data.apply(lambda x : self.trees[iter].predict_instance(x), axis=1)

        data['sum_exp'] = data.apply(lambda x:
                                     sum([np.exp(x['f_' + i + '_' + str(iter)]) for i in self.classes]), axis=1)

        for class_name in self.classes:
            proba_name = 'predict_proba_' + class_name
            f_m_name = 'f_' + class_name + '_' + str(iter)
            data[proba_name] = data.apply(lambda x: np.exp(x[f_m_name]) / x['sum_exp'], axis=1)
            
        data['predict_label'] = data.apply(lambda x: self._get_multi_label(x), axis=1)


    def _get_multi_label(self, x):
        label = None
        max_proba = -1
        for class_name in self.classes:
            if x['predict_proba_' + class_name] > max_proba:
                max_proba = x['predict_proba_' + class_name]
                label = class_name
        return label

    def get_gbdt_array(self):
        """
        Encode the GBDT by (feature, feature_val) --> key 
        """
        pass
    
    def get_gbdt_node_array(self):
        """
        Encode the GBDT by node --> Class Node()
        """
        pass