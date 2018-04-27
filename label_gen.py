# -*- coding: utf-8 -*-
import csv
import random

data_entry_dir = './dataset/Data_Entry_2017.csv'
index_dist_dir = './dataset/label_index.csv'
train_val_list_dir = './dataset/train_val_list.txt'
train_list_dir = './dataset/train_list.txt'
val_list_dir = './dataset/val_list.txt'

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

dist_scv_col = [
    'FileName',
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia'
]

if __name__ == '__main__':
    # Generate label index csv file
    with open(data_entry_dir) as f:
        with open(index_dist_dir, 'w+', newline='') as wf:
            #read in the lines
            writer = csv.writer(wf)
            lines = f.read().splitlines()
            del lines[0]
            writer.writerow(dist_scv_col)
            line_number = len(lines)

            #parse the file
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
                output = []
                output.append(file_name)
                output += vector
                writer.writerow(output)

    print("Label index generated at '%s'"%(index_dist_dir))
    
    # Split training list and  validation list
    with open(train_val_list_dir) as f:
        with open(train_list_dir, 'w+') as wf_t:
            with open(val_list_dir, 'w+') as wf_v:
                image_name_list = f.read().split('\n')
                random.shuffle(image_name_list)
                for i in range(len(image_name_list)):
                    if i < len(image_name_list)*(7/8):
                        wf_t.write(image_name_list[i]+'\n')
                    else:
                        wf_v.write(image_name_list[i]+'\n')

    print("Training list generated at '%s'"%(train_list_dir))
    print("Validation list generated at '%s'"%(val_list_dir))