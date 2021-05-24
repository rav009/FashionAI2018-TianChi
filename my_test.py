# coding: utf-8
import my_customdataset


a = my_customdataset.my_custom_dataset( \
	'/home/rav009/Desktop/Projects/FashionAI2018-TianChi/train_valid_allset/collar_design_labels/train/' \
	,'/home/rav009/Desktop/Projects/FashionAI2018-TianChi/label_master.csv')

print(len(a))