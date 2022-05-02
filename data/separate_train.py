import pandas as pd
import numpy as np
import csv

df = pd.read_csv("annotation_all_imgs_checked.csv", header=0)
cl_new_arr = df.loc[:, "classification_new"]
image_name_arr = df.loc[:, "new_file_name"]

""""
# for the following column: 
# 0: normal
# - 1: normal, spine
# - 2: normal, jaw
# - 11: normal, artefact
# - 10: normal, artefact?
# - 8: abnormal
"""

# prepare csv files to write
# open the files in the write mode:
f_train = open("all_images_train.csv", 'w')
f_test = open("all_images_test.csv", 'w')

# create the csv writers:
writer_train = csv.writer(f_train)
writer_test = csv.writer(f_test)

# add column names:
writer_test.writerow(['', 'new_file_name', 'orig_file_name', 'normal', 'anormal', 'artefacts', 'jaw', 'spine', 'tooth',
                      'dentition', 'comments', 'machine', 'classification_new'])
writer_train.writerow(['', 'new_file_name', 'orig_file_name', 'normal', 'anormal', 'artefacts', 'jaw', 'spine', 'tooth',
                      'dentition', 'comments', 'machine', 'classification_new'])

# these don't belong to the training set:
artefacts = [8, 10, 11]

# take the corresponding ones to the test set:
for i in range(0, len(df), 2):
    if cl_new_arr[i] in artefacts or cl_new_arr[i + 1] in artefacts:
        writer_test.writerow(df.iloc[i, :])
        writer_test.writerow(df.iloc[i + 1, :])

    else:
        writer_train.writerow(df.iloc[i, :])
        writer_train.writerow(df.iloc[i + 1, :])

# close the files:
f_train.close()
f_test.close()

# get test images' list:
df_test = pd.read_csv("all_images_test.csv")
test_size = len(df_test)

# get training images' list:
df_train = pd.read_csv("all_images_train.csv")

# open the new files in the write mode:
f_train_select = open("all_images_train_select.csv", 'w')
f_test_aug = open("all_images_test_aug.csv", 'w')

# create the csv writers for them:
writer_train_select = csv.writer(f_train_select)
writer_test_aug = csv.writer(f_test_aug)

# add column names:
writer_test_aug.writerow(['', 'new_file_name', 'orig_file_name', 'normal', 'anormal', 'artefacts', 'jaw', 'spine', 'tooth',
                         'dentition', 'comments', 'machine', 'classification_new'])
writer_train_select.writerow(['', 'new_file_name', 'orig_file_name', 'normal', 'anormal', 'artefacts', 'jaw', 'spine', 'tooth',
                             'dentition', 'comments', 'machine', 'classification_new'])

# select from original training set randomly half as many rows as in the original test set:
df_train_select = df_train.sample(n=int(test_size / 2))

# image names in the original training set:
image_name_arr_train = df_train.loc[:, "new_file_name"]

# take rows from the training set into the test set if these were selected:
for i in range(0, len(df_train), 2):
    if image_name_arr_train[i] in df_train_select.loc[:, "new_file_name"].values \
            or image_name_arr_train[i + 1] in df_train_select.loc[:, "new_file_name"].values:
        writer_test_aug.writerow(df_train.iloc[i, :])
        writer_test_aug.writerow(df_train.iloc[i + 1, :])
    else:
        writer_train_select.writerow(df_train.iloc[i, :])
        writer_train_select.writerow(df_train.iloc[i + 1, :])

# take the ones from the original test set:
for i in range(len(df)):
    if image_name_arr[i] in df_test.loc[:, "new_file_name"].values:
        writer_test_aug.writerow(df.iloc[i, :])

# close the files:
f_train_select.close()
f_test_aug.close()

# control:
df_test_aug = pd.read_csv("all_images_test_aug.csv")
test_size_aug = len(df_test_aug)
print("test_size_aug", test_size_aug)

df_train_selected = pd.read_csv("all_images_train_select.csv")
train_size_sel = len(df_train_selected)
print("train_size_sel", train_size_sel)

print("original_train_size", len(df_train))
print("original_test_size", test_size)





