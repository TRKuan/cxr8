# -*- coding: utf-8 -*-
import csv
import random

source_path = "./Data_Entry_2017.csv"
dist_path_train = "./Train_Label.csv"
dist_path_val = "./Val_Label.csv"
dist_path_test = "./Test_Label.csv"


disease_categories = {
        'Atelectasis': 0,
        'Cardiomegaly': 1,
        'Effusion': 2,
        'Infiltration': 3,
        'Mass': 4,
        'Nodule': 5,
        'Pneumonia': 6,
        'Pneumothorax': 7,
        'Consolidation': 8,
        'Edema': 9,
        'Emphysema': 10,
        'Fibrosis': 11,
        'Pleural_Thickening': 12,
        'Hernia': 13,
        }

if __name__ == '__main__':
    with open(source_path) as f:
        with open(dist_path_train, "w+", newline='') as wf_train:
            with open(dist_path_val, "w+", newline='') as wf_val:
                with open(dist_path_test, "w+", newline='') as wf_test:
                    writer_train = csv.writer(wf_train)
                    writer_val = csv.writer(wf_val)
                    writer_test = csv.writer(wf_test)
                    lines = f.read().splitlines()
                    #lines = lines[0:50]
                    del lines[0]
                    col = [
                            'FileName',
                            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
                            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                            'Pleural_Thickening', 'Hernia'
                            ]
                    writer_train.writerow(col)
                    writer_val.writerow(col)
                    writer_test.writerow(col)
                    random.shuffle(lines)
                    line_number = len(lines)
                    for i in range(line_number):
                        split = lines[i].split(',')
                        file_name = split[0]
                        label_string = split[1]
                        labels = label_string.split('|')
                        vector = [0 for _ in range(14)]
                        for label in labels:
                            if label != "No Finding":
                                vector[disease_categories[label]] = 1
                        vector.insert(0, file_name)
                        if i <= line_number*0.7:
                            writer_train.writerow(vector)#70%
                        elif i > line_number*0.7 and i <= line_number*0.8:
                            writer_val.writerow(vector)#10%
                        else :
                            writer_test.writerow(vector)#20%
    print("Label data generated")
                    
