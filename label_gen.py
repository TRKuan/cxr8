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
                    #read in the lines
                    writer_train = csv.writer(wf_train)
                    writer_val = csv.writer(wf_val)
                    writer_test = csv.writer(wf_test)
                    lines = f.read().splitlines()
                    #lines = lines[0:1000]#test with small data
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
                    line_number = len(lines)

                    #parse the file and generate a list fo patients
                    patients = []
                    for i in range(line_number):
                        split = lines[i].split(',')
                        file_name = split[0]
                        patient_id = int(split[3])-1
                        label_string = split[1]
                        labels = label_string.split('|')
                        vector = [0 for _ in range(14)]
                        for label in labels:
                            if label != "No Finding":
                                vector[disease_categories[label]] = 1
                        if len(patients) >= patient_id+1:
                            patients[patient_id]['file_name_list'].append(file_name)
                            patients[patient_id]['label_list'].append(vector)
                            patients[patient_id]['num'] += 1
                        else:
                            file_name_list = [file_name]
                            label_list = [vector]
                            num = 1
                            patients.append({
                                'file_name_list': file_name_list,
                                'label_list': label_list,
                                'num': num
                            })

                    #random generate labels
                    random.shuffle(patients)
                    train_list = []
                    val_list = []
                    test_list = []
                    for patient in patients:
                        out = []
                        for i in range(patient['num']):
                            out.append([])
                            out[i].append(patient['file_name_list'][i])
                            out[i] += patient['label_list'][i]
                        if len(train_list) < line_number*0.7:
                            train_list += out
                        elif len(val_list) < line_number*0.1:
                            val_list += out
                        else:
                            test_list += out

                    random.shuffle(train_list)
                    random.shuffle(val_list)
                    random.shuffle(test_list)

                    #save in files
                    for out in train_list:
                        writer_train.writerow(out)
                    for out in val_list:
                        writer_val.writerow(out)
                    for out in test_list:
                        writer_test.writerow(out)
    print("Label data generated")
    print("Train: {}\nValidation: {}\nTest: {}".format(len(train_list), len(val_list), len(test_list)))

