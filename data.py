import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

class DataSet:

    def __init__(self, df, standardize = False):
        """Data preprocessing"""

        #Make default col name of regression and binary classification called label
        label_name = df.columns[-1]
        if label_name != "label":
            self.df = df.rename({label_name : 'label'}, axis=1)
        else:
            self.df = df

        #For binary classification, use 0 as neg sample, 1 as pos sample
        if len(pd.unique(self.df['label'])) == 2:
            self.df['label'] = self.df['label'].apply(lambda x : int(0) if x <= 0 else int(1))
        
        #Nomalized the feature
        if standardize:
            self.df.iloc[:,0:-1] = scaler.fit_transform(self.df.iloc[:,0:-1].to_numpy())
            print("Nomalized!")

        self.df = shuffle(self.df)
        self.columns = self.df.columns[:-1]


    def _encodeTable(self):
        """Make an hash table to bind the (feature, feature value)"""

        #Extract all unique feature values
        self.unique_values = [(col, list(pd.unique(self.df[col]))) for col in self.df.columns[:self.df.columns.get_loc('label')]]
        #Endcode the unique feature and feature values by (feature, feature value)
        self.merge_values = [(e[0], i) for e in self.unique_values for i in e[1]]
        self.lookup_tables = {i: self.merge_values[i] for i in range(len(self.merge_values))}

    def getData(self):
        return self.df

    def getTrainData(self):
        return self.df[:int(self.df.shape[0] * 0.8)]
    
    def getTestData(self):
        return self.df[int(self.df.shape[0] * 0.8):]
    
    def getLength(self):
        return len(self.df)

    def getRandomElements(self, drops):
        """Return a random (feature, feature value)"""
        columns = self.columns
        columns = columns.drop(drops)
        if columns.size == 0:
            #This is the case if max_depth > Num of features, the columns will be 0 after .drop() 
            f = self.columns[random.randint(0, self.columns.size - 1)]
        else:
            f = columns[random.randint(0, columns.size - 1)]

        fval =  self.df[f][random.randint(0, self.df.shape[0] - 1)]
        return (f, fval)

    def getRandomElement(self):
        """Return a random (feature, feature value)"""

        # randomly pick up a feature
        f = self.columns[random.randint(0, self.columns.size - 1)]
        # randomly pick up a feature value belongs to the feature
        fval =  self.df[f][random.randint(0, self.df.shape[0] - 1)]
        
        return (f, fval)
        #return self.merge_values[random.randint(0,len(self.merge_values) - 1)]

    def getIndex(self, value):
        """Map the (feature, feature value) to a key value"""
        return self.merge_values.index(value)

    def getLookupTable(self):
        return self.lookup_tables
    
    def getMergeValues(self):
        return self.merge_values

    def isRepeat(self):
        return len(set(self.merge_values)) == len(self.merge_values)

    def getReaminData(self, indexs):
        return self.df.loc[indexs]
