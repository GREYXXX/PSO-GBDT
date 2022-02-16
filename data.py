import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

class DataSet:

    def __init__(self, train, test, target_name, standardize=False):
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

    # def __init__(self, x_train, y_train, x_test, y_test):
    #     self.x_train = x_train
    #     self.y_train = y_train
    #     self.x_test = x_test
    #     self.y_test = y_test
   
    #     # For binary classification, use 0 as neg sample, 1 as pos sample for the y_train
    #     label_name = y_train.columns[0]
    #     if len(pd.unique(self.y_train[label_name])) and len(pd.unique(self.y_test[label_name]))== 2:
    #         self.y_train[label_name] = self.y_train[label_name].apply(lambda x : int(0) if x <= 0 else int(1))
    #         self.y_test[label_name]  = self.y_test [label_name].apply(lambda x : int(0) if x <= 0 else int(1))
        
    #     self.columns = self.x_train.columns


    def encode_table(self):
        """Make an hash table to bind the (feature, feature value)"""

        #Extract all unique feature values
        self.unique_values = [(col, list(pd.unique(self.train[col]))) for col in self.train.columns[:self.train.columns.get_loc('label')]]
        #Endcode the unique feature and feature values by (feature, feature value)
        self.merge_values = [(e[0], i) for e in self.unique_values for i in e[1]]
        self.lookup_tables = {i: self.merge_values[i] for i in range(len(self.merge_values))}

    def get_data(self):
        df = pd.concat([self.train, self.test],axis=0,ignore_index=True)
        return df

    def get_train_data(self):
        return self.train
    
    def get_test_data(self):
        return self.test

    def get_random_elements(self, drops):
        """Return a random (feature, feature value)"""
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
        #return self.merge_values[random.randint(0,len(self.merge_values) - 1)]

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
