# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:42:34 2022

@author: sergio
"""

import numpy as np 

def organiza_datos(classification_report,normalfloat,tumorfloat,accuracy):

    a=classification_report

    b=a.split('\n')[2].split('      ')
    c=a.split('\n')[3].split('      ')
    d=a.split('\n')[5].split('      ')


    normalfloat = np.append(normalfloat, [b[2],b[3],b[4]])
    tumorfloat = np.append(tumorfloat, [c[2],c[3],c[4]])
    accuracy= np.append(accuracy, d[4])
    


    return normalfloat,tumorfloat,accuracy



