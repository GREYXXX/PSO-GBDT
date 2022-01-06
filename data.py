import pandas as pd
import random

class DataSet:

    def __init__(self, df):
        self.df = df

    def _encodeTable(self):
        self.unique_values = [(col, list(pd.unique(self.df[col]))) for col in self.df.columns[:-self.df.columns.get_loc('label')]]
        self.merge_values = [(e[0], i) for e in self.unique_values for i in e[1]]
        self.lookup_tables = {i: self.merge_values[i] for i in range(len(self.merge_values))}

    def getData(self):
        return self.df
    
    def getLength(self):
        return len(self.df)

    def getRandomElement(self):
        return self.merge_values[random.randint(0,len(self.merge_values) - 1)]

    def getIndex(self, value):
        return self.merge_values.index(value)

    def getLookupTable(self):
        return self.lookup_tables
    
    def getMergeValues(self):
        return self.merge_value

    def isRepeat(self):
        return len(set(self.merge_values)) == len(self.merge_values)

    def getReaminData(self, indexs):
        return self.df.loc[indexs]