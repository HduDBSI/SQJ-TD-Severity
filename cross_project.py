#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cross_project_SITD-SSI.py
@Contact :   sichengluis@gmail.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/12/31 9:22   Sichengluis      1.0         None
'''
import pandas as pd
import numpy as np
from mlxtend.classifier import StackingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn import under_sampling, combine

# import os
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

def preprocess_df(df,train=False):
    df = df.sample(frac=1,random_state=42)
    y = df['label'].values
    method_vec_unified = np.array(list(map(strvec2arr, df['method_code'].values)))


    metrics = df[['loc','par_nbr','comment_nbr','expr_stmt_nbr','name_exprs_nbr','variable_declaration_exprs_nbr','method_call_nbr',
                       'loopQty','assignmentsQty','mathOperationsQty','stringLiteralsQty','numbersQty'
                       ]].values

    mms = MinMaxScaler()


    SEMANTIC_PERCENTAGE=0.7

    method_vec_unified = SEMANTIC_PERCENTAGE * method_vec_unified
    metrics=(1-SEMANTIC_PERCENTAGE)*metrics

    x = np.concatenate((method_vec_unified, metrics), axis=1)

    # x=mms.fit_transform(x)

    if train==True:
        sampling=combine.SMOTEENN(random_state=42,
                         smote=SMOTE(sampling_strategy=0.3,random_state=42),
                         enn=under_sampling.EditedNearestNeighbours(sampling_strategy="auto",
                                                                    kind_sel="mode"))
        x, y = sampling.fit_resample(x, y)
        print(len(x))
        return x, y
    else:
        print(len(x))
        return x,y

def cross_project(model,file_name):
    res_table = []
    project_name_list = ['Ant','ArgoUML','Columba','Hibernate','JEdit','JFreeChart','JMeter','JRuby','SQuirrel']

    proj_len = len(project_name_list)

    total_pr = 0
    total_rc = 0
    total_f1 = 0

    for proj in project_name_list:
        print(proj)
        PREPROCESSED_FILE='./td-severity-final-preprocess.csv'
        df = pd.read_csv(PREPROCESSED_FILE,encoding="utf_8_sig")

        train_df = df[df['project'] != proj]
        train_df.reset_index(inplace=True,drop=True)
        test_df = df[df['project'] == proj]
        test_df.reset_index(inplace=True,drop=True)
        train_x,train_y = preprocess_df(train_df,True)
        test_x,test_y = preprocess_df(test_df)

        model.fit(train_x,train_y)
        pred_test_y = model.predict(test_x)

        test_precision = precision_score(test_y,pred_test_y)
        test_recall = recall_score(test_y,pred_test_y)
        test_f1 = f1_score(test_y,pred_test_y)
        total_pr += test_precision
        total_rc += test_recall
        total_f1 += test_f1
        print("Test:precision:{:.4f}, recall:{:.4f}, f1:{:.4f}".format(test_precision,test_recall,test_f1))
        row = [proj,round(test_precision,4),round(test_recall,4),round(test_f1,4)]
        res_table.append(row)
    res_row = ["average",round(total_pr / proj_len,4),round(total_rc / proj_len,4),round(total_f1 / proj_len,4)]
    res_table.append(res_row)
    print("Avg:precision:{:.4f}, recall:{:.4f}, f1:{:.4f}".format(round(total_pr / proj_len,4),round(total_rc / proj_len,4),round(total_f1 / proj_len,4)))
    res_df = pd.DataFrame(res_table,columns=['project','precision','recall','f1'])
    res_df.to_csv(file_name,index=None,encoding="utf_8_sig")

if __name__ == '__main__':
    SVM = svm.SVC(C=0.8,random_state=42)
    RF = RandomForestClassifier(criterion='gini',min_samples_split=3,min_samples_leaf=1,random_state=42,max_depth=72,
                                n_estimators=78,n_jobs=-1)
    XGB = XGBClassifier(n_estimators=105,max_depth=58,colsample_bytree=0.9,learning_rate=0.24210526315789474)
    Stacking_Classifier = StackingClassifier(classifiers=[XGB,RF],meta_classifier=SVM,use_probas=True)
    Stacking_file = "Cross.csv"
    best_model = Stacking_Classifier
    cross_project(best_model,Stacking_file)







