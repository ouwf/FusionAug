# FusionAug - PyTorch 

 This source code shows how to train a feature embedding network to perform finger vein-based biometric verification, based on the paper "Fusion Loss and Inter-class Data Augmentation for Deep Finger Vein Feature Learning". Our real-time demo: https://www.youtube.com/watch?v=815MXXj2gtU.


### Instruction
1. Install the required libraries
   ```
   $ conda env create --name pt1.2 --file env.yml
   $ conda activate pt1.2
   ```

2. Download the preprocessed FV-USM finger vein database:
https://portland-my.sharepoint.com/:u:/g/personal/weifengou2-c_my_cityu_edu_hk/EYEw8-g0xZRKufJfYK9ZRwgBEtiw63dKzd60g-iJQZwpyA?e=fPITqi


3. Start training by specifiying the name of the database, the dataset location, network architecture, loss function, data augmentation options, etc. 

   #### Example 1: train with fusion loss using intra-class data augmentation and inter-class data augmentation via vertical (top-bottom) flipping.
   `python3 train.py --dataset FVUSM --data "path to your data" --pretrained --loss fusion --intra_aug --inter_aug "TB"`
   #### Example 2: simply run the shell script `./run.sh`


4. Evaluate the verification performance of the provided checkpoint on the FVUSM testing set
   
   `python3 -u ./eval.py --ckpt "ckpt path" --dataset FVUSM --data "dataset path" --network resnet18`

### Dependencies
Python3.6, PyTorch 1.2, torchvision 0.4.0

### Note
The copyright of the FV-USM database is owned by Dr. Bakhtiar Affendi Rosdi,
School of Electrical and Electronic Engineering, USM (http://drfendi.com/fv_usm_database/). We provide the preprocessed database here only for testing the source code. If any copyright issues are violated, please inform us to remove the database.  

### Citation
[1] Ou Weifeng, Po Laiman, Zhou Chang, Rehman Yasar Abbas Ur, Xian Pengfei, Zhang Yujia. Fusion Loss and Inter-class Data Augmentation for Deep Finger Vein Feature Learning [J]. Expert Systems with Applications, 2021, 171(7):114584. https://doi.org/10.1016/j.eswa.2021.114584.

[2] Mohd Shahrimie Mohd Asaari, Shahrel A. Suandi, Bakhtiar Affendi Rosdi, Fusion of Band Limited Phase Only Correlation and Width Centroid Contour Distance for finger based biometrics, Expert Systems with Applications, Volume 41, Issue 7, 1 June 2014, Pages 3367-3382, ISSN 0957-4174, http://dx.doi.org/10.1016/j.eswa.2013.11.033.
