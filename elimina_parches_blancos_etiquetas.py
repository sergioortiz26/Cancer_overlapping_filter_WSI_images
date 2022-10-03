# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:55:27 2022

@author: sergio
"""

import numpy as np

def elimina_parches_blancos_etiquetas(x_cada_parche,etiquetas_pred_binarias,etiquetas_reales):
    
    etiquetas_pred_sin_blancos=[]
    etiquetas_reales_sin_blancos=[]
    
    ii=0
    
    for i in range(len(x_cada_parche)):
                    
        media_path=int(np.mean(x_cada_parche[i])*255)     
                
        if media_path<240:
                   
            etiquetas_pred_sin_blancos.append(etiquetas_pred_binarias[i])
            etiquetas_reales_sin_blancos.append(etiquetas_reales[i])
                        
            ii+1   
        
    return etiquetas_pred_sin_blancos,etiquetas_reales_sin_blancos