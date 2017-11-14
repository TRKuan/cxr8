# Chest X-Ray


Deep learning practice on CXR dataset.

### Setup
1. Setup [pytorch](http://pytorch.org/) enviroment.
2. Download the dataset [here](https://nihcc.app.box.com/v/ChestXray-NIHCC).
3. Extract the images into directory `./images`.

To generate train, validation, and test data.

    python label_gen.py
 
This will generate 3 csv files: `Train_Label.csv`, `Val_Label.csv`, and `Test_Label.csv`.


----------------------------------------------
See the following paper for more information.  
Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, Ronald M. Summers. ChestX-ray8: [Hospital-scale Chest X-ray Database and Benchmarks on Weakly- Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/pdf/1705.02315.pdf), IEEE CVPR, pp. 3462-3471,2017
