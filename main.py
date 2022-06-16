import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import KNN

iris = load_iris()

X = iris.data[:, :]
y = iris.target
y_name = iris.target_names

testDatas = np.empty((0,4))
#testDatas 는 실제로 테스트 할때 사용 할 데이터 들 입니다.
testLabels = np.array([])
#testLabels 는 테스트 할 데이터들의 실제 label 값 들 입니다.

#testDatas 는 매 15번째 원소입니다.
for i in range(1, 11): 
    testDatas = np.append(testDatas, np.array([X[15*i - 1]]), axis = 0) #iris.data 의 매 15번째 원소를 testDatas 에 삽입합니다.
    testLabels = np.append(testLabels, y[15*i - 1]) #testDatas의 실제 lable을 추가해 줍니다.

for i in range(1, 11): 
    X = np.delete(X,15*i - i, axis = 0) #testDatas 들을 원본 data 에서 제거해 줍니다.
    y = np.delete(y,15*i - i, axis = 0) # testLabels 들을 원본 label 에서 제거해 줍니다

rows, columns = testDatas.shape # testDatas 의 행과 열 값 입니다.


#KNN 생성자의 parameter 
#첫번째 : K 를 결정
#두번째 : input iris datas
#세번째: input iris labels


knn = KNN.KNN(15, X, y)
print("----------------------------------Majority vote----------------------------------------------")
for i in range(0, rows):
    #Majority vote
    print("Test Data Index: ", i, " Computed class: ", y_name[knn.Obtain_majority_vote(testDatas[i])], ", True class: ", y_name[int(testLabels[i])])

print("----------------------------------Weighted Majority vote-------------------------------------")
for i in range(0, rows):
    #Weighted Majority vote
    print("Test Data Index: ", i, " Computed class: ", y_name[knn.weighted_majority_vote(testDatas[i])], ", True class: ", y_name[int(testLabels[i])])


"""
정확도 측정을 하기 위해서는 주석을 제거해 주세요
accuracy1 = [0 for i in range(0, 135)] # 정확도 측정을 위한 array
accuracy2 = [0 for i in range(0, 135)] # 정확도 측정을 위한 array
for j in range (1, 136):
    success = 0 
    knn = KNN.KNN(j, X, y)
    for i in range(0, rows):
        if(knn.Obtain_majority_vote(testDatas[i]) == int(testLabels[i])):
            success = success + 0.1
    accuracy1[j-1] = success

for j in range (1, 136):
    success = 0 
    knn = KNN.KNN(j, X, y)
    for i in range(0, rows):
        if(knn.weighted_majority_vote(testDatas[i]) == int(testLabels[i])):
            success = success + 0.1
    accuracy2[j-1] = success

k1 = [i for i in range(1, 136)]
k2 = [i for i in range(1, 136)]
plt.plot(k1, accuracy1, label = "majority vote")
plt.plot(k2, accuracy2, linestyle="--", label = "weighted majority vote")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
"""