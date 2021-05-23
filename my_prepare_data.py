# coding: utf-8

import os, shutil, random

# Read label.csv
# For each task, make folders, and copy picture to corresponding folders
# This is renjie modification version, which was adpated to the new data folder schema in the longterm competition.

label_train_dir1 = '/home/rav009/Projects/FashionAI2018-TianChi/fashionAI_attributes_train1/Annotations/label.csv'
label_train_dir2 = '/home/rav009/Projects/FashionAI2018-TianChi/fashionAI_attributes_train2/Annotations/label.csv'


label_dict = {'coat_length_labels': [],     # 衣长设计
              'lapel_design_labels': [],    # 翻领设计
              'neckline_design_labels': [],  # 颈线设计　
              'skirt_length_labels': [],     # 裙长设计
              'collar_design_labels': [],    # 领子设计
              'neck_design_labels': [],      # 脖颈设计
              'pant_length_labels': [],      # 裤长设计
              'sleeve_length_labels': []}    # 袖长设计


task_list = label_dict.keys()

def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))

# Images/collar_design_labels/4d8a38b29930a403e5e2167c6e2327b5.jpg, collar_design_labels, nnynn

all_lable = {}
with open(label_train_dir1, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        if path in all_lable:
            continue
        else:
            all_lable.setdefault(path,[])
            path1='/home/rav009/Projects/FashionAI2018-TianChi/fashionAI_attributes_train1/'+path
            label_dict[task].append((path1, label))

with open(label_train_dir2, 'r') as f:
    lines = f.readlines()
    tokens = [l.rstrip().split(',') for l in lines]
    for path, task, label in tokens:
        if path in all_lable:
            continue
        else:
            all_lable.setdefault(path,[])
            path1='/home/rav009/Projects/FashionAI2018-TianChi/fashionAI_attributes_train2/'+path
            label_dict[task].append((path1, label))


mkdir_if_not_exist(['train_valid_allset'])

for task, path_label in label_dict.items(): 
    mkdir_if_not_exist(['train_valid_allset',  task])
    train_count = 0 # 对每一类都要重新置0
    n = len(path_label) # 每个task有多少条数据
    m = len(list(path_label[0][1])) # 每个task有几类

    for mm in range(m):
        mkdir_if_not_exist(['train_valid_allset', task, 'train', str(mm)])
        mkdir_if_not_exist(['train_valid_allset', task, 'val', str(mm)])
        
    random.seed(2021)
    random.shuffle(path_label)
    for path, label in path_label:
        label_index = list(label).index('y')
        # if 'm' in label:    # 如果存在m标签,就寻找m标签
        #     m_index = list(label).index('m')
        #     label_index = label_index if random.randint(1,5)>2 else m_index # 60%选择y对应的label,40%选择m对应的label
        src_path = os.path.join(path)
        if train_count < n * 0.95:
            shutil.copy(src_path,
                        os.path.join('train_valid_allset', task, 'train', str(label_index)))
        else:
            shutil.copy(src_path,
                        os.path.join('train_valid_allset', task, 'val', str(label_index)))
        train_count += 1
    print( ' Finished ' + task)
print( ' All finished!')