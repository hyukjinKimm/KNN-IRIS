import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class KNN:
    def __init__(self, K, Datas, Labels): #생성자 함수 입니다.
        self.k = K
        self.datas = Datas
        self.labels = Labels
    
    def get_dist(self, b, c): # 두 점의 거리를 계산하는 함수 입니다.
        dist = np.linalg.norm(b-c)
        
        return dist

    def get_K_neighbor(self, input):
        rows = self.datas.shape[0] # training data 의 갯수
        distances = np.array([self.get_dist(self.datas[i, :], input) for i in range(rows)]) # dataset 과 input data 의 거리 구하기
       
        sorted_Distances = np.sort(distances)
        candidates = np.array([np.where(distances == sorted_Distances[i])[0][0] for i in range(self.k)])
        candidates = candidates[:self.k]

        return candidates, distances
    
    def m_vote(self, input):
        result = np.array([0, 0, 0])
        candidates = self.get_K_neighbor(input)[0]
        for i in range(self.k):
            result[self.labels[candidates[i]]] += 1
        
        return result.argmax()

    def w_vote(self, input):
        result = np.array([0, 0, 0], dtype = 'float64')
        candidates, distances = self.get_K_neighbor(input)
        distances = distances / distances.max() + 1
    
        for i in range(self.k):
            result[self.labels[candidates[i]]] +=  1/distances[i]
        
        return result.argmax()