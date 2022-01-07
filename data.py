import pandas as pd
import random

class DataSet:

    def __init__(self, df):
        self.df = df
        self.columns = self.df.columns[:-1]

    def _encodeTable(self):
        #Extract all unique feature values
        self.unique_values = [(col, list(pd.unique(self.df[col]))) for col in self.df.columns[:self.df.columns.get_loc('label')]]
        #Endcode the unique feature and feature values by (feature, feature value)
        self.merge_values = [(e[0], i) for e in self.unique_values for i in e[1]]
        self.lookup_tables = {i: self.merge_values[i] for i in range(len(self.merge_values))}

    def getData(self):
        return self.df
    
    def getLength(self):
        return len(self.df)

    def getRandomElement(self):
        # randomly pick up a feature
        f = self.columns[:-1][random.randint(0, self.columns.size - 2)]
        # randomly pick up a feature value belongs to the feature
        fval =  self.df[f][random.randint(0, self.df.shape[0] - 1)]
        
        return (f, fval)
        #return self.merge_values[random.randint(0,len(self.merge_values) - 1)]

    def getIndex(self, value):
        return self.merge_values.index(value)

    def getLookupTable(self):
        return self.lookup_tables
    
    def getMergeValues(self):
        return self.merge_values

    def isRepeat(self):
        return len(set(self.merge_values)) == len(self.merge_values)

    def getReaminData(self, indexs):
        return self.df.loc[indexs]
