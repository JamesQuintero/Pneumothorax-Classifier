# Pneumothorax-Classifier

Takes in chest radiographs with labels of -1 for no pneumothorax, and values for a mask of the pneumothorax, and it uses different model architectures to determine if out of sample chest radiographs contain pneumothorax. 

Current techniques: 
* Filters and crops radiograph images to focus on ribcage
* Image augmentation
* Train dataset split by 50% positive and 50% negative
* resampling ensemble
* Architectures include CNN and U-net
* Classification type icnludes binary and segmentation


This project is being designed to be somewhat easy to add in a new type of radiograph and require minimal changes to the code. 
