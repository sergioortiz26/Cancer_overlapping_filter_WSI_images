# Cancer_overlapping_filter_WSI_images

Code for different filters in the test phase to improve label assignment by predicting a CNN


Whole slide image (WSI) and the tumor mask associated with the normal tissue (zone black) and the tumor tissue (zone white). 

----------------------------------
CREATES PATCHES OF THE IMAGES WSI

Throught crea_parches_test_overlapping, crea_parches_test_sin_overlapping and crea_parches_train_and_validation are create the different patches asociated to the images. 
  - Crea_parches_train_and_validation: Create patches asociated with train and validate phase. 
  - crea_parches_test_sin_overlapping: Create patches associated with the test phase, patches that do not overlap with each other.
  - crea_parches_test_overlapping: Create patches associated with the test phase, patches that overlap with each other.
  
   To corroborate that each image has been well divided into patches, you can use the script comprobacion_mascaras.py, where selecting the path of the patches of an image, you can redo them to check their correct division into patches. 
   Note that this is not done with the training and validation images because at this stage the background patches that do not correspond to the tissue have been removed.
   
-------------------------------------------------
