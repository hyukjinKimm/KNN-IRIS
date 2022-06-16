import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import KNN

iris = load_iris()

X = iris.data[:, :].astype(np.float64)
y = iris.target
y_name = iris.target_names

# Every 15-th data for test
testDatas = np.array([X[15*i + 14] for i in range(10)])
testLabels = np.array([y[15*i + 14] for i in range(10)])

# Delete 15-th data from dataset
X = np.delete(X, [15*i + 14 for i in range(10)], axis = 0)
y = np.delete(y, [15*i + 14 for i in range(10)], axis = 0)


# K 는 7
knn = KNN.KNN(7, X, y)
rows = testDatas.shape[0] # testDatas 의 개수

print("----------------------------------Majority vote----------------------------------------------")
for i in range(0, rows):
    #Majority vote
    print("Test Data Index: ", i, " Computed class: ", y_name[knn.m_vote(testDatas[i])], ", True class: ", y_name[int(testLabels[i])])
print('\n')
print("----------------------------------Weighted Majority vote-------------------------------------")
for i in range(0, rows):
    #Weighted Majority vote
    print("Test Data Index: ", i, " Computed class: ", y_name[knn.w_vote(testDatas[i])], ", True class: ", y_name[int(testLabels[i])])
print('\n')


m_acc = [0 for i in range(0, 100)] # 정확도 측정을 위한 array
w_acc = [0 for i in range(0, 100)] # 정확도 측정을 위한 array

# k는 1부터 100 까지
for k in range (1, 101):
    success = 0 
    knn = KNN.KNN(k, X, y)
    for i in range(rows):
        if(knn.m_vote(testDatas[i]) == int(testLabels[i])):
            success += 1/rows
    m_acc[k-1] = success

for k in range (1, 101):
    success = 0 
    knn = KNN.KNN(k, X, y)
    for i in range(rows):
        if(knn.w_vote(testDatas[i]) == int(testLabels[i])):
            success += 1/rows
    w_acc[k-1] = success

k1 = [i for i in range(1, 101)]
k2 = [i for i in range(1, 101)]
plt.plot(k1, m_acc, label = "majority vote")
plt.plot(k2, w_acc, linestyle="--", label = "weighted majority vote")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
