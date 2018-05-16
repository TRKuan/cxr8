# Chest X-Ray

Detect common thoracic diseases from chest X-ray images.

### Setup
1. Setup [pytorch](http://pytorch.org/)(0.4.0) enviroment.
2. Download the dataset [here](https://nihcc.app.box.com/v/ChestXray-NIHCC).
3. Extract the images into directory `dataset/images`.
4. Put `Data_Entry_2017.csv`, `BBox_List_2017.csv`, `test_list.txt`, and `train_val_list.txt` into directory `dataset`.

To generate train, validation, and test data entry.

    python label_gen.py
 
This will separate `train_val_list.txt` into `train_list.txt` and `val_list.txt`.  
3 csv files `train_label.csv`, `val_label.csv`, and `test_label.csv` will be generated as data entry.

### Training
To train the model, You may modify the hyperparameters in the file and run

    python train.py

To start tensorboard

    tensorboard --logdir=./runs


----------------------------------------------
See the following paper for more information about the dataset.  
Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers. ChestX-ray8: [Hospital-scale Chest X-ray Database and Benchmarks on Weakly- Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/pdf/1705.02315.pdf), IEEE CVPR, pp. 3462-3471,2017
