#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DANDA.py
@Contact :   sichengluis@gmail.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/24 14:49   Sichengluis      1.0         None
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from my_cross_validate import *
# from preprocess_dataset import *
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from mlxtend.classifier import StackingClassifier
from mlxtend.feature_selection import ColumnSelector




import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def strvec2arr(str_val):
    '''
    将字符串形式的向量转成向量列表
    :param str_val:字符串形式的向量
    :return:向量列表
    '''
    str_list = str_val.split(',')
    float_list=[]
    for item in str_list:
        res=float(item)
        float_list.append(res)

    return float_list


def within_both(model,res_file_name):
    res_table = []
    project_name_list = ['Ant','ArgoUML','Columba','Hibernate','JEdit','JFreeChart','JMeter','JRuby','SQuirrel']
    proj_len=len(project_name_list)
    total_pr = 0
    total_rc = 0
    total_f1 = 0
    df=pd.read_csv("./td-severity-final-preprocess.csv")

    for proj in project_name_list:
        proj_df = df[df['project'] == proj]

        method_vec_unified = np.array(list(map(strvec2arr,proj_df['method_code'].values)))

        mms = MinMaxScaler()
        method_vec_unified = mms.fit_transform(method_vec_unified)
        metrics = proj_df[
            ['loc','par_nbr','comment_nbr','expr_stmt_nbr','name_exprs_nbr','variable_declaration_exprs_nbr','method_call_nbr',
             'loopQty','assignmentsQty','mathOperationsQty','stringLiteralsQty','numbersQty'
             ]].values
        metrics = mms.fit_transform(metrics)
        SEMANTIC_PERCENTAGE = 0.7

        method_vec_unified = SEMANTIC_PERCENTAGE * method_vec_unified
        metrics = (1 - SEMANTIC_PERCENTAGE) * metrics

        x = np.concatenate((method_vec_unified,metrics),axis=1)
        y = np.array(proj_df['label'].values)

        print(proj)
        print(len(x))
        precision,recall,f1 = my_cross_validate(model,x,y)
        total_pr += precision
        total_rc += recall
        total_f1 += f1

        res_row = [proj,round(precision,4),round(recall,4),round(f1,4)]
        res_table.append(res_row)
    res_row = ["average",round(total_pr / proj_len,4),round(total_rc / proj_len,4),round(total_f1 / proj_len,4)]
    res_table.append(res_row)
    df = pd.DataFrame(res_table,columns=['project','precision','recall','f1'])
    df.to_csv(res_file_name,index=None,encoding="utf_8_sig")



if __name__ == '__main__':
    RF = RandomForestClassifier(criterion='gini',min_samples_split=2,min_samples_leaf=1,random_state=42,max_depth=72,
                                n_estimators=98,n_jobs=-1)
    SVM = svm.SVC(C=0.8,random_state=42)
    LR = LogisticRegression()

    XGB = XGBClassifier(n_estimators=105,max_depth=78,colsample_bytree=0.9,learning_rate=0.06)

    Stacking_Classifier = StackingClassifier(classifiers=[XGB,RF],meta_classifier=SVM)
    Stacking_file = "Within.csv"

    within_both(Stacking_Classifier,Stacking_file)
