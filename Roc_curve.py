# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:52:46 2022

@author: sergio


"""


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot


def curve_auc(label_true,prediction):
    
    auc = roc_auc_score(label_true, prediction[:,1])
    
    
    ns_fpr, ns_tpr, _ = roc_curve(label_true, prediction[:,1])
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Curve AUC')
    pyplot.xlabel('Tasa de Falsos Positivos')
    pyplot.ylabel('Tasa de Verdaderos Positivos')
    pyplot.legend()
    pyplot.show()
    
    return auc