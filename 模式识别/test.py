'''
Author: 雨冷那听听
Date: 2023-11-02 12:22:15
LastEditTime: 2023-11-08 10:56:52
LastEditors: 雨冷那听听
Description: 
FilePath: \HyperspectralSegmentation\test.py
还请老师手下留情，多少给点分吧~
'''
import numpy as np
import scipy.io as scio
import random
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

def KL_transform(train_datas, test_datas, target_dim):
    """
    使用K-L变换进行特征降维
    :param X: 输入数据，形状为(n_samples, n_features)
    :param target_dim: 目标维度
    :return: 降维后的数据，形状为(n_samples, target_dim)
    """
    kpc = KernelPCA(n_components=target_dim, kernel='rbf', gamma=10)
    reduced_data = kpc.fit_transform(train_datas)
    test_return = kpc.transform(test_datas)
    return reduced_data, test_return
# PCA降维
def PCA_reduction(train_datas, test_datas, target_dim):
    pca = PCA(n_components=target_dim)
    reduced_data = pca.fit_transform(train_datas)
    test_return = pca.transform(test_datas)
    return reduced_data, test_return

# K邻域分类
def classify_data_knn(train_datas, test_datas, train_labels, test_labels, target_k):
    classifier = KNeighborsClassifier(n_neighbors=target_k)
    classifier.fit(train_datas, train_labels)
    accuracy = classifier.score(test_datas, test_labels)
    return accuracy

# SVM分类
def classify_data_svm(train_datas, test_datas, train_labels, test_labels):

    classifier = SVC()
    classifier.fit(train_datas, train_labels)
    accuracy = classifier.score(test_datas, test_labels)
    return accuracy

# 神经网络分类，采用MLP
def classify_data_ann(train_datas, test_datas, train_labels, test_labels, hidden_layer_size=(64,)):
    classifier = MLPClassifier(solver='adam', alpha=0.0001, max_iter=500, hidden_layer_sizes=hidden_layer_size, random_state=None)
    classifier.fit(train_datas, train_labels)
    accuracy = classifier.score(test_datas, test_labels)
    return accuracy

number_classes = 16                                 # 类别数（不含背景类）
rate = 0.7                                          # 训练数据占比

# 读取训练数据
data_path = './data/Indian_pines_corrected.mat'
label_path = './data/Indian_pines_gt.mat'
data = scio.loadmat(data_path)['indian_pines_corrected'].reshape(-1, 200)
label = scio.loadmat(label_path)['indian_pines_gt'].flatten()

# 统计各类像素的数据
count = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[]}
for i in range(label.shape[0]):
    if label[i] != 0:                               # 排除背景类
        count[label[i]-1].append(data[i])

# 打乱数据
for key in count:
    random.shuffle(count[key])

# 构造训练集与测试集 
train_datas = []
train_labels = []
test_datas = []
test_labels = []
count_train = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[]}

number_max_train = 0                                # 统计数据量最多的类别的样本数，作为其他类别数据扩充的标准
# 获取基本的训练集和测试集
for key in count:
    train_num = int(count[key].__len__() * rate)    # 获取当前类中用于训练的样本数
    if number_max_train < train_num:
        number_max_train = train_num
    for j in range(count[key].__len__()):           # 将当前类别的样本拆分到训练集与测试集
        if j < train_num:
            count_train[key].append(count[key][j])
        else:
            test_datas.append(count[key][j])
            test_labels.append(key)

# 训练集数据扩充（重复）
for key in count_train:
    number_temp_train = count_train[key].__len__()
    for i in range(number_max_train - number_temp_train):
        count_train[key].append(count_train[key][i%number_temp_train])

# 交替放置各类数据（shuffle）
for i in range(number_max_train):
    for key in range(number_classes):
        train_datas.append(count_train[key][i])
        train_labels.append(key)

train_datas = np.array(train_datas, dtype='float32')
train_labels = np.array(train_labels)
test_datas = np.array(test_datas, dtype='float32')
test_labels = np.array(test_labels)

# 数据特征内归一化（样本间同一位置的特征归一化）
mean = train_datas.mean(axis=0)
std = train_datas.std(axis=0)
train_datas = (train_datas - mean) / std
test_datas = (test_datas - mean) / std


# train_datas, test_datas = KL_transform(train_datas, test_datas,10)
# test_datas = PCA_reduction(test_datas)[1]

# train_datas = KL_transform(train_datas, 10)[0]
# test_datas = KL_transform(test_datas, 10)[1]
# # 使用PCA进行特征降维
# from sklearn.decomposition import PCA
# pca = PCA(n_components=50)
# train_datas = pca.fit_transform(train_datas)
# test_datas = pca.transform(test_datas)

# print("---KL+SVM---")
# train_datas, test_datas = KL_transform(train_datas, test_datas, 16)
# accuracy = classify_data_svm(train_datas, test_datas, train_labels, test_labels)
# print("Accuracy:", accuracy)

print("---PCA+SVM---")
pca_train, pca_test= PCA_reduction(train_datas, test_datas, 25)
accuracy = classify_data_svm(pca_train, pca_test, train_labels, test_labels)
print("Accuracy:", accuracy)
# plt.scatter(pca_train[:, 0], pca_train[:, 1], c=train_labels)
# plt.show()

print("---PCA+Knn---")
# train_datas, test_datas = PCA_reduction(train_datas, test_datas, 25)
accuracy = classify_data_knn(pca_train, pca_test, train_labels, test_labels, 5)
print("Accuracy:", accuracy)

print("---PCA+MLP---")
# train_datas, test_datas = PCA_reduction(train_datas, test_datas, 16)
accuracy = classify_data_ann(pca_train, pca_test, train_labels, test_labels,)
print("Accuracy:", accuracy)
# # 使用SVM进行分类
# from sklearn import svm
# clf = svm.SVC()
# clf.fit(train_datas, train_labels)
# accuracy = clf.score(test_datas, test_labels)
# print("Accuracy:", accuracy)