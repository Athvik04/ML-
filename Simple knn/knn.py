import numpy as np
import csv

class KNN():
    def __init__(self,dataset,data_uji,k):
        ''' class constructor '''
        self.ddir = dataset
        self.uji = self.read_dataset(data_uji)
        self.uji = self.cleaning_dataset(self.uji)
        self.__k__ = k
        self.data = self.read_dataset(self.ddir)
        self.data = self.cleaning_dataset(self.data)
        self.all_euclidean = self.get_euclidean(self.data, self.uji, self.__k__)
        self.predict_result = self.init_predict(self.all_euclidean, self.__k__)

    def read_dataset(self,ddir):
        ''' methods to read data to numpy array '''
        with open(ddir,'r') as fi:
            reader = csv.reader(fi)
            data = [item for item in reader]
            data = np.asarray(data)
            return data

    def cleaning_dataset(self, data):
        ''' methods to clean data to numpy array '''
        label = (data[:,-1] == 'good').astype(int)
        data[:,-1] = label
        return data.astype(float)

    def get_euclidean(self,data,uji,k):
        ''' methods to calculate euclidean distance'''
        uji_atr = uji[0,:-1]
        data_without_label = data[:,:-1]
        uji_repeat = np.tile(uji_atr, data_without_label.shape[0]).reshape((data_without_label.shape[0],data_without_label.shape[1]))
        _euclidean = np.sqrt(np.sum(np.square(abs(data_without_label-uji_repeat)), axis=1))
        _euclidean = np.array([_euclidean]).reshape((data_without_label.shape[0], 1))
        _sorted = np.argsort(_euclidean,kind='quicksort', axis=0)
        _euclidean = np.concatenate((_euclidean, _sorted),axis=1)
        all_euclidean = np.concatenate((data, _euclidean),axis=1)
        return all_euclidean

        # select value that less than equal to k
        
    def init_predict(self, all_, k):
        ''' initiating prediction ''' 
        nearest_value = all_[np.where(all_[:,-2]<=k)]
        # returning frequent value
        get_column = nearest_value[:,-3].astype(int)
        return np.bincount(get_column).argmax()
        
    def get_predict_result(self):
        ''' returning prediction result '''
        if self.predict_result == 1:
            return 'good'
        else:
            return 'bad'
