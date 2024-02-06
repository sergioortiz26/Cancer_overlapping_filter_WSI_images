

import cv2 
import matplotlib 
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

path='D:/Sergio/Patch/Data_6_4/Patch_100x100_test/12880_15_porc/'          #path to an image patch folder           


#check for each image and its mask which area has and does not have camcer to label it.

Pos_x=[]
Pos_y=[]

Pos_x2=[]
Pos_y2=[]

etiquetas=[]


b=pd.DataFrame(glob.glob(path+'*.png'))

Pos_x_2=np.zeros


for i in glob.glob(path+'*.png'):
    
    name=str(i)
    save_name = name.split('/')[-1]   #split()method splits a string into a list
    
    posicion_y=save_name.split('.')[1]
    posicion_x=save_name.split('.')[2]
    
    etiqueta=save_name.split('.')[3].split('.')[0]
        
    Pos_x.append(int(posicion_x))
    Pos_y.append(int(posicion_y))
    #print(i)
    

    
max_x=int(max(Pos_x))
max_y=int(max(Pos_y))



imagen2=np.ones(((max_y+1)*100,(max_x+1)*100,3))
x=100

for i in tqdm(range(max_y+1)):
    for j in range(max_x+1):
        
        
        Pos_x2.append(int(i))
        Pos_y2.append(int(j))
        
        for k in glob.glob(path+'*.png'):

            name=str(k)
            save_name = name.split('/')[-1]  
            label=save_name.split('.')[3]
            posicion_y=int(save_name.split('.')[1])
            posicion_x=int(save_name.split('.')[2])
            etiqueta=save_name.split('.')[3]
            
            if posicion_y==i and posicion_x==j:
                print(1)

                if label=='tumor':
                    etiquetas.append(int(1))
                    imagen2[i*x:(i+1)*x,j*x:(j+1)*x,:]=255
                else: 
                    imagen2[i*x:(i+1)*x,j*x:(j+1)*x,:]=0
                    etiquetas.append(int(0))

            
            imagen2=imagen2.astype(np.uint8)    
                        
fig, ax = plt.subplots()
ax.scatter(Pos_x, Pos_y, c=etiquetas, cmap="coolwarm", s=20);

matplotlib.pyplot.imshow(imagen2)



import cv2
import numpy as np


img_mask=cv2.imread('D:/Sergio/Patch/Data_6_4/HUP_masks/12880_annotation_mask.png')     # mask to an image patch folder 
img1=img_mask[0:(max_y+1)*100,0:(max_x+1)*100,:]


img_=cv2.imread('D:/Sergio/Patch/Data_6_4/Imagenes_test/12880_idx5.png')          #  image patch folder 
img_rgb = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
img2=img_rgb[0:(max_y+1)*100,0:(max_x+1)*100,:]


dst = cv2.addWeighted(img1,0.7,imagen2,0.3,0)

img3=imagen2/2
img3=img3.astype(np.uint8)

imgadd = cv2.add(img1,img2)
imgadd2 = cv2.add(img2,img3)

matplotlib.pyplot.imshow(imgadd)
matplotlib.pyplot.imshow(imgadd2)
matplotlib.pyplot.imshow(dst)


