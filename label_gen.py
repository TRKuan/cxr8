# -*- coding: utf-8 -*-
import csv
import random

source_dir = './dataset/Data_Entry_2017.csv'
dist_dir = './dataset/label_index.csv'

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
    with open(source_dir) as f:
        with open(dist_dir, 'w+', newline='') as wf:
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

    print("Label index generated at '%s'"%(dist_dir))
