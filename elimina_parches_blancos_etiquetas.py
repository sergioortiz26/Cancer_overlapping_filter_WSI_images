# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:55:27 2022

@author: sergio
"""

import numpy as np

#The images may have patches that are white because they do not have tissue,
#simply by obtaining the image of the sample there is a part that does not give information about the background. 
#These patches can be removed to avoid labelling with patches that contain no information.  
#What is done is to set a threshold and if the mean colour of the patch is lower than that threshold, 
# (i.e. the area being studied is white, that patch is removed from the data set to train the CNN).

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
