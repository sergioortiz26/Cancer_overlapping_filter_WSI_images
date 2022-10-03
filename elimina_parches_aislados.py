# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 17:28:22 2022

@author: sergio
"""

import numpy as np



def elimina_parches_aislados(imagen):
    
    
    imagen_modificada=np.copy(imagen)

    
    for i in range(np.size(imagen,0)):   # i recorrera las filas 
        for j in range(np.size(imagen,1)):    # j recorrera las filas 
     
    
            if i==0:
                if j==0:    
                    if (imagen[i][j]==0 and imagen[i+1][j]==1 and imagen[i+1][j+1]==1 and imagen[i][j+1]==1):
                        imagen_modificada[i][j]=1
                    if (imagen[i][j]==1 and imagen[i+1][j]==0 and imagen[i+1][j+1]==0 and imagen[i][j+1]==0):
                        imagen_modificada[i][j]=0
                elif j!=0 and j!= np.size(imagen,1)-1: 
                    if (imagen[i][j]==0 and imagen[i][j+1]==1 and imagen[i+1][j+1]==1 and imagen[i+1][j-1]==1 and imagen[i][j-1]==1 and imagen[i+1][j]==1):
                        imagen_modificada[i][j]=1
                    if (imagen[i][j]==1 and imagen[i][j+1]==0 and imagen[i+1][j+1]==0 and imagen[i+1][j-1]==0 and imagen[i][j-1]==1 and imagen[i+1][j]==0):
                        imagen_modificada[i][j]=0
                elif j==np.size(imagen,1): 
                    if (imagen[i][j]==0  and imagen[i+1][j-1]==1 and imagen[i][j-1]==1 and imagen[i+1][j]==1):
                        imagen_modificada[i][j]=1
                    if (imagen[i][j]==1  and imagen[i-1][j-1]==0 and imagen[i][j-1]==0 and imagen[i-1][j]==0):
                        imagen_modificada[i][j]=0
                        
              
                        
            elif i==np.size(imagen,0)-1:
                if j==0:    
                    if (imagen[i][j]==0 and imagen[i][j+1]==1 and imagen[i-1][j]==1 and imagen[i-1][j+1]==1):
                        imagen_modificada[i][j]=1
                    if (imagen[i][j]==1 and imagen[i][j+1]==0 and imagen[i-1][j]==0 and imagen[i-1][j+1]==0):
                        imagen_modificada[i][j]=0
                elif j!=0 and j!= np.size(imagen,1)-1: 
                    if (imagen[i][j]==0 and imagen[i-1][j]==1 and imagen[i-1][j-1]==1 and imagen[i-1][j+1]==1 and imagen[i][j-1]==1 and imagen[i][j-1]==1):
                        imagen_modificada[i][j]=1
                    if (imagen[i][j]==1 and imagen[i-1][j]==0 and imagen[i-1][j-1]==0 and imagen[i-1][j+1]==0 and imagen[i][j-1]==0 and imagen[i][j-1]==0):
                        imagen_modificada[i][j]=0
                elif j==np.size(imagen,0): 
                    if (imagen[i][j]==0  and imagen[i][j-1]==1 and imagen[i-1][j-1]==1 and imagen[i-1][j]==1):
                        imagen_modificada[i][j]=1
                    if (imagen[i][j]==1  and imagen[i][j-1]==0 and imagen[i-1][j-1]==0 and imagen[i-1][j]==0):
                        imagen_modificada[i][j]=0
                        
                        
            elif j==0:
                
                if i!=0 and i!=np.size(imagen,0)-1:
                    if (imagen[i][j]==0 and imagen[i][j+1]==1 and imagen[i+1][j+1]==1 and imagen[i+1][j]==1 and imagen[i-1][j]==1 and imagen[i-1][j+1]==1):
                        imagen_modificada[i][j]=1
                    if (imagen[i][j]==1 and imagen[i][j+1]==0 and imagen[i+1][j+1]==0 and imagen[i+1][j]==0 and imagen[i-1][j]==0 and imagen[i-1][j+1]==0):
                        imagen_modificada[i][j]=0
            
                        
            elif j==np.size(imagen,1)-1:
                
                if i!=0 and i!=np.size(imagen,0)-1: 
                    if (imagen[i][j]==0 and imagen[i][j-1]==1 and imagen[i+1][j-1]==1 and imagen[i+1][j]==1 and imagen[i-1][j]==1 and imagen[i-1][j-1]==1):
                        imagen_modificada[i][j]=1
                    if (imagen[i][j]==1 and imagen[i][j-1]==0 and imagen[i+1][j-1]==0 and imagen[i+1][j]==0 and imagen[i-1][j]==0 and imagen[i-1][j-1]==0):
                        imagen_modificada[i][j]=0
                        
            else:

                
                if (imagen[i][j]==0 and imagen[i][j+1]==1 and imagen[i+1][j+1]==1 and imagen[i+1][j]==1 and imagen[i+1][j-1]==1 and imagen[i][j-1]==1 and imagen[i-1][j-1]==1 and imagen[i-1][j]==1 and imagen[i-1][j+1]==1):
                    imagen_modificada[i][j]=1
                if (imagen[i][j]==1 and imagen[i][j+1]==0 and imagen[i+1][j+1]==0 and imagen[i+1][j]==0 and imagen[i+1][j-1]==0 and imagen[i][j-1]==0 and imagen[i-1][j-1]==0 and imagen[i-1][j]==0 and imagen[i-1][j+1]==0):
                    imagen_modificada[i][j]=0


    return imagen_modificada