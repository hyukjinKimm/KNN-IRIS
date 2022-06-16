import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class KNN:
    def __init__(self, K, Datas, Labels): #생성자 함수 입니다.
        self.k = K
        self.datas = Datas
        self.labels = Labels

    def CalCulate_distance(a, b, c): # 두 점의 거리를 계산하는 함수 입니다.
        dist = np.linalg.norm(b-c)
        
        return dist

    def Obtain_KNN_Neighbor(self, target):
        dists = [] # datas 들과 target과의 거리가 들어 갈 array 입니다.
        candidates = [] #k 개의 제일 가까운 data 들의  index가 들어 갈 array 입니다.
        rows, columns = self.datas.shape # datas 들의 행과 열 값 입니다.
        for i in range(0, rows):
            dists.append(self.CalCulate_distance(self.datas[i, :], target))
        origin = dists[:] #정렬 전의 dists 값을 origin 에 저장 해놓습니다.
        dists.sort() #dists 를 오름차순 으로 정렬합니다.
        for i in range(0,self.k):
            candidates.append(origin.index(dists[i])) #가장 거리가 가까운 k 개의 원소들의 index를 추출하여 candidates 에 저장합니다.
        return candidates, origin #candidates 와 원본 거리값 array 인 origin 을 리턴합니다.

    def Obtain_majority_vote(self, target):
        near_neibors_labels = [] # candidates 들에 해당하는 label 값이 들어 갈 배열입니다.
        candidates , dists= self.Obtain_KNN_Neighbor(target) #target 에서 k개의 가까운 이웃들의 index를 구합니다.
        for i in range(0, len(candidates)):
            near_neibors_labels.append(self.labels[candidates[i]]) #candidates 들의 label 값을 추출합니다.
        max_num = max(set(near_neibors_labels), key = near_neibors_labels.count) #다수결로 target의 label 을 정합니다.
        return max_num # 결정된 label 값을 리턴합니다

    def weighted_majority_vote(self, target):
        near_neibors_labels = [] # candidates 들에 해당하는 label 값이 들어 갈 배열입니다.
        candidates , dists= self.Obtain_KNN_Neighbor(target) #target 에서 k개의 가까운 이웃들의 인덱스를 구합니다.
        weight = [0, 0, 0] # 각 label 의 weitgh 가 들어갈 array 입니다.
        for i in range(0, len(candidates)):
            if dists[candidates[i]] < 0.01 : # 무한대를 방지하기위해 0.01 보다 작다면 일정한 weight 를 줍니다.
                weight[self.labels[candidates[i]]] =  weight[self.labels[candidates[i]]] + 100
            else:# 그렇지 않다면 distance 의 역수를 weight 로 줍니다.
                weight[self.labels[candidates[i]]] =  weight[self.labels[candidates[i]]] + 1/dists[candidates[i]]
        
        return weight.index(max(weight)) #weight array 에서 가장 값이 큰 element 를 갖고 있는 index 를 리턴합니다.
            
    
        

    
    
        

