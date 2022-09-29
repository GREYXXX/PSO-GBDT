# @Author XI RAO
# CITS4001 Research Project

from typing import List, Tuple
import pandas as pd
import random
from sklearn.utils import shuffle
from make_bins import GetBins
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataSet:

    def __init__(
        self, 
        X : pd.DataFrame, 
        y : pd.Series,  
        max_bins : int = -1,
        use_validation: bool = True,
        )-> None:

        if use_validation:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
            X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size = 0.5, random_state=42)
            self.validation =  pd.concat([X_val, y_val], axis = 1).rename(columns = {y.name : 'label'})
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.train = pd.concat([X_train, y_train], axis = 1).rename(columns = {y.name : 'label'})
        self.test = pd.concat([X_test, y_test], axis = 1).rename(columns = {y.name : 'label'})
        self.df = pd.concat([X, y], axis = 1).rename(columns = {y.name : 'label'})
        
        self.use_validation = use_validation
        self.max_bins = max_bins
        self.bins = None

        self.columns = X.columns[:]

    def encode_table(
        self, 
        use_pretrain : bool = False, 
        internal_splits : List[Tuple[str, float]] = []
        )-> None:
        """Make an hash table to bind the (feature, feature value)"""

        if use_pretrain:
            self.merge_values = [i for i in internal_splits]
        else:
            if self.max_bins > 0:
                print(f"use bin with {self.max_bins}")
                bins = GetBins(
                    self.df, 
                    self.columns, 
                    max_bin = self.max_bins,
                )
                bins = {i : bins[i][1:-1] for i in bins}
                self.bins = bins
                self.unique_values = [(col, bins[col]) for col in self.columns]
            else:
                self.unique_values = [(col, pd.unique(self.df[col])) for col in self.columns]
            
            self.merge_values = [(e[0], i) for e in self.unique_values for i in e[1]] 

        self.lookup_tables = {i: self.merge_values[i] for i in range(len(self.merge_values))}

    def get_train_data(self):
        return self.train.copy()

    def get_test_data(self):
        return self.test.copy()

    def get_fitness_test_data(self):
        if self.use_validation:
            return self.validation
        else:
            return self.train

    def get_random_elements(self, drops):
        """Return a random (feature, feature value)"""

        if self.max_bins > 0 and self.bins != None:
            columns = [i for i in self.bins if len(self.bins[i]) > 0]
            f = columns[random.randint(0, len(columns) - 1)]
            fval = self.bins[f][random.randint(0, len(self.bins[f]) - 1)]
            return (f, fval)

        else:
            columns = self.columns
            columns = columns.drop(drops)
            if columns.size == 0:  
                #This is the case if max_depth > Num of features, the columns will be 0 after .drop() 
                f = self.columns[random.randint(0, self.columns.size - 1)]        
            else:
                f = columns[random.randint(0, columns.size - 1)]

            fval =  self.df[f][random.randint(0, self.df.shape[0] - 1)]
            return (f, fval)

    def get_random_element(self):
        """Return a random (feature, feature value)"""

        # randomly pick up a feature
        f = self.columns[random.randint(0, self.columns.size - 1)]

        # randomly pick up a feature value belongs to the feature
        fval =  self.df[f][random.randint(0, self.df.shape[0] - 1)]
        
        return (f, fval)

    def get_index(self, value):
        """Map the (feature, feature value) to a key value"""
        return self.merge_values.index(value)

    def get_lookup_table(self):
        return self.lookup_tables
    
    def get_merge_values(self):
        return self.merge_values

    def is_repeat(self):
        return len(set(self.merge_values)) == len(self.merge_values)