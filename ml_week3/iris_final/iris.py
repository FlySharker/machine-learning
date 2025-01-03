#!/usr/bin/env python
# -*- coding: UTF-8 -*-


__author__ = "Maylon"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import graphviz
from sklearn import datasets
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def Bunch2dataframe(sklearn_dataset):
    """
    将sklearn数据集Bunch类型转成DataFrame
    :param sklearn_dataset: sklearn中的数据集
    :return: 处理后的dataframe，最后一列为标签列
    """
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)        # 追加一列标签列
    return df

def Draw_tree(clf, filename, feature_names=None, class_names=None):
    """
    绘制决策树并保存为*.pdf文件
    :param clf: 训练后的模型
    :param filename: 保存的文件名
    :param feature_names: 特征名
    :param class_names: 标签名
    :return: None
    """
    dot_data = tree.export_graphviz(clf,
                                    out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render(filename)
    print("Done.")


def best_depth_tree(train, test):
    """
    调参得到最佳的max_depth值并返回对应训练后的模型
    :param train: 训练集
    :param test: 测试集
    :return: 训练后的模型列表和测试集预测准确率最大值的索引
    """
    train_score_list = []
    test_score_list = []
    clf_list = []
    max_test_depth = 10     # 最大树深(超参数上限)
    train_data = train.iloc[:, :-1]
    train_target = train.iloc[:, -1]
    test_data = test.iloc[:, :-1]
    test_target = test.iloc[:, -1]
    for i in range(max_test_depth):
        clf = DecisionTreeClassifier(criterion="gini",
                                     max_depth=i+1,
                                     random_state=30,
                                     splitter="best"
                                     )
        clf = clf.fit(train_data, train_target)     # 训练模型
        score_train = clf.score(train_data, train_target)       # 训练集预测准确率
        score = clf.score(test_data, test_target)       # 测试集预测准确率
        train_score_list.append(score_train)
        test_score_list.append(score)
        clf_list.append(clf)
    plt.xlabel('树深')
    plt.ylabel('得分')
    plt.title('决策树')
    plt.plot(range(1, max_test_depth+1), train_score_list, color="blue", label="train")        # 绘制分数曲线
    plt.plot(range(1, max_test_depth+1), test_score_list, color="red", label="test")
    plt.legend()
    plt.show()
    print('训练集预测准确率:',train_score_list)
    print('测试集预测准确率:', test_score_list)
    print('测试集最大准确率:',max(test_score_list))
    return clf_list, test_score_list.index(max(test_score_list))


def sklearn():
    data = datasets.load_iris()     # 加载数据集
    dataset = Bunch2dataframe(data)     # 转换成dataframe类型进行处理，最后一列为标签列
    print('原始数据集:\n',dataset)
    train, test = train_test_split(dataset)     # 切分训练集和测试集
    feature_names = dataset.columns[:-1]        # 获取特征名
    clf_list, i = best_depth_tree(train, test)      # 训练模型
    print("max_depth: " + str(i+1))
    clf = clf_list[i]     # 选取测试集预测准确率最大值的模型
    Draw_tree(clf, "iris", feature_names=feature_names)     # 绘制决策树


if __name__ == '__main__':
    sklearn()