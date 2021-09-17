# FusionAug - PyTorch 

 This source code shows how to train a feature embedding network to perform finger vein-based biometric verification, based on the paper "Fusion Loss and Inter-class Data Augmentation for Deep Finger Vein Feature Learning". Our real-time demo: https://www.youtube.com/watch?v=815MXXj2gtU.


### Instruction
1. Download the preprocessed FV-USM finger vein database:
https://portland-my.sharepoint.com/:u:/r/personal/weifengou2-c_my_cityu_edu_hk/Documents/FingerVeinDataset/public%20databases/FVUSM/FV-USM-processed.tgz?csf=1&web=1&e=EM2B6b


2. Run train.py by specifiying the name of the database, the dataset location, network architecture, loss function, data augmentation options, etc. 

   #### Example 1: train with fusion loss using intra-class data augmentation and inter-class data augmentation via vertical (top-bottom) flipping.
   `python3 train.py --dataset FVUSM --data "path to your data" --pretrained --fusion --intra_aug --inter_aug "TB"`
   #### Example 2: simply run the shell script `./run.sh`

### Dependencies
Python3.6, PyTorch 1.2 

### Note
The copyright of the FV-USM database is owned by Dr. Bakhtiar Affendi Rosdi,
School of Electrical and Electronic Engineering, USM (http://drfendi.com/fv_usm_database/). We provide the preprocessed database here only for testing the source code. If any copyright issues are violated, please inform us to remove the database.  

### Citation
[1] Ou Weifeng, Po Laiman, Zhou Chang, Rehman Yasar Abbas Ur, Xian Pengfei, Zhang Yujia. Fusion Loss and Inter-class Data Augmentation for Deep Finger Vein Feature Learning [J]. Expert Systems with Applications, 2021, 171(7):114584. https://doi.org/10.1016/j.eswa.2021.114584.

[2] Mohd Shahrimie Mohd Asaari, Shahrel A. Suandi, Bakhtiar Affendi Rosdi, Fusion of Band Limited Phase Only Correlation and Width Centroid Contour Distance for finger based biometrics, Expert Systems with Applications, Volume 41, Issue 7, 1 June 2014, Pages 3367-3382, ISSN 0957-4174, http://dx.doi.org/10.1016/j.eswa.2013.11.033.
