# @Author XI RAO
# CITS4001 Research Project

import pandas as pd
import random
from sklearn.utils import shuffle
from make_bins import GetBins
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

class DataSet:

    def __init__(self, train, test, target_name, standardize=False, is_bin = False):
        """Data preprocessing"""

        #Make default col name of regression and binary classification called label
        label_name =target_name
        if label_name != "label":
            self.train = train.rename({label_name : 'label'}, axis=1)
            self.test  = test.rename({label_name : 'label'}, axis=1)
        else:
            self.train = train
            self.test  = test

        #For binary classification, use 0 as neg sample, 1 as pos sample
        if len(pd.unique(self.train['label'])) == 2:
            self.train['label'] = self.train['label'].apply(lambda x : int(0) if x <= 0 else int(1))
            self.test['label']  = self.test['label'].apply(lambda x : int(0) if x <= 0 else int(1))
        
        #Nomalized the feature
        if standardize:
            self.train.iloc[:,0:-1] = scaler.fit_transform(self.df.iloc[:,0:-1].to_numpy())
            self.test.iloc[:,0:-1]  = scaler.fit_transform(self.df.iloc[:,0:-1].to_numpy())
            print("Nomalized!")

        self.columns = self.train.columns[:-1]
        self.df = pd.concat([self.train, self.test],axis=0,ignore_index=True)
        self.is_bin = is_bin


    def encode_table(self, pretrained = False, pretrain_arrays = []):
        """Make an hash table to bind the (feature, feature value)"""

        if pretrained:
            self.merge_values = [i for i in pretrain_arrays]
        else:
            if self.is_bin:
                bins = GetBins(self.train, self.columns, max_bin = 100)
                bins = {i : bins[i][1:-1] for i in bins}
                self.bins = bins
                self.unique_values = [(col, bins[col]) for col in self.train.columns[:self.train.columns.get_loc('label')]]
            else:
                self.unique_values = [(col, list(pd.unique(self.train[col]))) 
                                                for col in self.train.columns[:self.train.columns.get_loc('label')]]
            
            self.merge_values = [(e[0], i) for e in self.unique_values for i in e[1]] 

        self.lookup_tables = {i: self.merge_values[i] for i in range(len(self.merge_values))}

    def get_data(self):
        return self.df.copy()

    def get_train_data(self):
        return self.train.copy()
    
    def get_test_data(self):
        return self.test.copy()

    def get_random_elements(self, drops):
        """Return a random (feature, feature value)"""

        if self.is_bin:
            columns = [i for i in self.bins if len(self.bins[i]) > 0]
            f = columns[random.randint(0, len(columns) - 1)]
            fval =  self.bins[f][random.randint(0, len(self.bins[f]) - 1)]
            return (f, fval)

        else:
            columns = self.columns
            columns = columns.drop(drops)
            if columns.size == 0:  
                #This is the case if max_depth > Num of features, the columns will be 0 after .drop() 
                f = self.columns[random.randint(0, self.columns.size - 1)]        
            else:
                f = columns[random.randint(0, columns.size - 1)]

            fval =  self.train[f][random.randint(0, self.train.shape[0] - 1)]
            return (f, fval)

    def get_random_element(self):
        """Return a random (feature, feature value)"""

        # randomly pick up a feature
        f = self.columns[random.randint(0, self.columns.size - 1)]

        # randomly pick up a feature value belongs to the feature
        fval =  self.train[f][random.randint(0, self.train.shape[0] - 1)]
        
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

    def get_reamin_data(self, indexs):
        return self.df.loc[indexs]
