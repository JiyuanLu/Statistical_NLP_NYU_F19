import numpy as np

class DataLoader(object):
    def __init__(self):
        self.X = []
        self.Y = []
        self.unique_chars = []
        self.unique_classes = []
        self.num_of_classes = 0
        self.num_of_datapoints = 0
        
    def load(self, path):
        with open(path, 'r', encoding='iso-8859-1') as t:
            for line in t:
                record = line.strip('\n').split('\t')
                self.Y.append(record[0])
                self.X.append(record[1])   
        self.unique_chars = sorted(list({l for word in self.X for l in word}))
        self.unique_classes = sorted(list(set(self.Y)))
        self.num_of_classes = len(set(self.Y))
        self.num_of_datapoints = len(self.Y)
        return self.X, self.Y
        