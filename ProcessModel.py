import imp
import pickle
import numpy as np

class PreprocessXgbModel:
    def __init__(self, file):
        self.model = pickle.load(open(file, "rb"))
        self.n_trees = self.model.n_estimators
        self.max_depth = self.model.max_depth
        self.learning_rate = self.model.learning_rate
        self.tree_df = self.model.get_booster().trees_to_dataframe()
        self.internal_nodes = [(self.tree_df['Feature'].values[i], self.tree_df['Split'].values[i]) 
                                    for i in range(len(self.tree_df)) if self.tree_df['Feature'].values[i] != 'Leaf']

        self.tree_df['Yes'] = self.tree_df['Yes'].fillna('0')
        self.tree_df['No'] = self.tree_df['No'].fillna('0')
        self.df_table = {i : self.tree_df[self.tree_df['Tree'] == i] for i in range(self.n_trees)}

    
    def get_model(self):
        return self.model 

    def get_n_trees(self):
        return self.n_trees
    
    def get_max_depth(self):
        return self.max_depth 
    
    def get_learning_rate(self):
        return self.learning_rate
    
    def get_internal_splits(self):
        return self.internal_nodes
        
    def get_tree_arrays(self):
        return {i : self.__get_tree_array(self.df_table[i]) for i in range(self.n_trees)}

    def get_internal_tree_arrays(self):
        return {i : self.__get_internal_tree_array(self.df_table[i]) for i in range(self.n_trees)}

    def __get_tree_infos(self, df):
        split = df['Split'].values
        feature = df['Feature'].values
        child = []
        merge = []
        for i in range(len(split)):
            if np.isnan(split[i]):
                merge.append(df['Gain'].values[i])
            else:
                merge.append(split[i])
    
        for i in range(len(df)):
            if df['Yes'].values[i] != '0' and df['No'].values[i] != '0':
                v1 = int(df['Yes'].values[i].split('-')[1])
                v2 = int(df['No'].values[i].split('-')[1])
                child.append((v1, v2))
            else:
                child.append(('0', '0'))

        return feature, merge, child
    
    def __get_internal_tree_array(self, df):
        feature, merge, child = self.__get_tree_infos(df)
        trees_array = [((i,feature[i]), merge[i], child[i]) for i in range(len(df)) if feature[i] != "Leaf"]
        return trees_array
    
    def __get_tree_array(self, df):
        feature, merge, child = self.__get_tree_infos(df)
        trees_array = [(feature[i], merge[i], child[i]) for i in range(len(df))]
        return trees_array




class PreprocessSklModel:
    def __init__(self, file):
        self.model = pickle.load(open(file, "rb"))
    
    def get_internal_splits(self, features_names):
        splits = []
        for i in range(self.model.estimators_.shape[0]):
            feature = self.model.estimators_[i][0].tree_.feature
            feature_val = self.model.estimators_[i][0].tree_.threshold
            for j in range(len(feature)):
                if feature[j] != -2:
                    splits.append((features_names[feature[j]], feature_val[j]))
        return splits