# -*- coding: utf-8 -*-
import csv
import os
import random

dataset_dir = './dataset'

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

dist_csv_col = [
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
    # Split training list and  validation list
    
    train_list = []
    val_list = []
    with open(os.path.join(dataset_dir, 'train_val_list.txt')) as f:
        with open(os.path.join(dataset_dir, 'train_list.txt'), 'w+') as wf_t:
            with open(os.path.join(dataset_dir, 'val_list.txt'), 'w+') as wf_v:
                image_name_list = f.read().split('\n')
                
                # group the same patients together to guaranty no patient overlap between splits
                patients = []
                last = ''
                for i in range(len(image_name_list)):
                    if last == image_name_list[i][:8]:
                        patients[-1].append(image_name_list[i])
                    else:
                        patients.append([image_name_list[i]])
                    last = image_name_list[i][:8]
                
                # shuffle
                random.shuffle(patients)
                
                # split them into train and val
                train_list = []
                val_list = []
                train_num = 0
                for i in range(len(patients)):
                    if train_num < len(image_name_list)*(7/8):
                        train_list += patients[i]
                        train_num += len(patients[i])
                    else:
                        val_list += patients[i]
                        
                # sort the list
                train_list.sort()
                val_list.sort()
                
                # write file
                for data in train_list[:-1]:
                    wf_t.write(data+'\n')
                wf_t.write(train_list[-1])
                for data in val_list[:-1]:
                    wf_v.write(data+'\n')
                wf_v.write(val_list[-1])

    print('Training list generated')
    print('Validation list generated')
    
    
    # Generate label index csv file
    
    f_de = open(os.path.join(dataset_dir, 'Data_Entry_2017.csv'))
    dataset_type = ['train', 'val', 'test']
    f = {t:open(os.path.join(dataset_dir, t+'_list.txt')) for t in dataset_type}
    wf = {t:open(os.path.join(dataset_dir, t+'_label.csv'), 'w+', newline='') for t in dataset_type}
    writer = {t:csv.writer(wf[t]) for t in dataset_type}
    # write header
    for t in dataset_type:
        writer[t].writerow(dist_csv_col)
    
    # read
    lines_de = f_de.read().splitlines()
    del lines_de[0]
    image_name_list = {t:f[t].read().split('\n') for t in dataset_type}

    #parse the file
    pos = {t:0 for t in dataset_type}
    for i in range(len(lines_de)):
        split = lines_de[i].split(',')
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
        
        # write to train, val, or test
        for t in dataset_type:
            if pos[t] >= len(image_name_list[t]):
                continue
            if file_name == image_name_list[t][pos[t]]:
                writer[t].writerow(output)
                pos[t] += 1
    
    f_de.close()
    for t in dataset_type:
        f[t].close()
        wf[t].close()

    print('Label index generated')
    
