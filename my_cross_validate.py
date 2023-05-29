#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cross_validate.py
@Contact :   sichengluis@gmail.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/4 15:23   Sichengluis      1.0         None
'''
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
import  numpy as np
import xgboost as xgb
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek
from imblearn import under_sampling, combine


def my_cross_validate(model,x,y):
    '''
    根据需求自己封装的交叉验证方法
    :param model: 模型
    :param x: 特征
    :param y: 标签
    :return:
    '''
    # 进行十折交叉验证
    sKFold = StratifiedKFold(n_splits=10, shuffle=True,random_state=28)
    precision_total, recall_total, f1_total,acc_total,auc_total = 0, 0, 0,0,0
    for train_index, test_index in sKFold.split(x, y):
        # 一次交叉验证
        train_x, test_x = x[train_index], x[test_index]
        train_y, test_y = y[train_index], y[test_index]

        # sampling=SMOTE(random_state=42)

        sampling = combine.SMOTEENN(random_state=21,
                                    smote=SMOTE(random_state=21),
                                    enn=under_sampling.EditedNearestNeighbours(sampling_strategy="auto",
                                                                               kind_sel="mode"))
        train_x,train_y = sampling.fit_resample(train_x,train_y)

        model.fit(train_x, train_y)
        pred_test_y = model.predict(test_x)
        pr_once=precision_score(test_y,pred_test_y)
        rc_once= recall_score(test_y, pred_test_y)
        f1_once=f1_score(test_y,pred_test_y)
        precision_total += pr_once
        recall_total +=rc_once
        f1_total += f1_once
    print("precision:{:.4f}, recall:{:.4f}, f1:{:.4f}".format(precision_total / 10,  recall_total /10, f1_total / 10))
    return round(precision_total / 10,4),round(recall_total /10,4),round(f1_total / 10,4)


