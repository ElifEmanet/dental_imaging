import pandas as pd
import csv

df = pd.read_csv("2022-07-08_dental_annotation_final.csv", header=0)
t_v_t_arr = df.loc[:, "train_val_test"]


""""
## 2022-07-08_dental_annotation_final.csv

### classification
0: normal
10: normal + artefact
8: anomal
11: anomal + artefact
7: anomal?

### view_classes_3

0 … normal
1 … spine
2 … jaw
3 … spine & jaw
"""

# prepare csv files to write
# open the files in the write mode:
f_train = open("new_all_images_train.csv", 'w')
f_val = open("new_all_images_val.csv", 'w')
f_test = open("new_all_images_test.csv", 'w')
f_train_clf = open("new_all_images_train_clf.csv", 'w')

# create the csv writers:
writer_train = csv.writer(f_train)
writer_val = csv.writer(f_val)
writer_test = csv.writer(f_test)
writer_train_clf = csv.writer(f_train_clf)

# add column names:
writer_train.writerow(['', 'file_name', 'machine', 'normal', 'anormal', 'artefacts', 'classification', 'view_classes_3',
                      'is_jaw', 'is_spine', 'split_anomaly', 'split_classification', 'train_val_test'])
writer_val.writerow(['', 'file_name', 'machine', 'normal', 'anormal', 'artefacts', 'classification', 'view_classes_3',
                    'is_jaw', 'is_spine', 'split_anomaly', 'split_classification', 'train_val_test'])
writer_test.writerow(['', 'file_name', 'machine', 'normal', 'anormal', 'artefacts', 'classification', 'view_classes_3',
                      'is_jaw', 'is_spine', 'split_anomaly', 'split_classification', 'train_val_test'])
writer_train_clf.writerow(['', 'file_name', 'machine', 'normal', 'anormal', 'artefacts', 'classification',
                           'view_classes_3', 'is_jaw', 'is_spine', 'split_anomaly', 'split_classification',
                           'train_val_test'])

# take the corresponding ones to the test set:
for i in range(len(df)):
    if t_v_t_arr[i] == "train":
        writer_train.writerow(df.iloc[i, :])
        writer_train_clf.writerow(df.iloc[i, :])

    elif t_v_t_arr[i] == "val":
        writer_val.writerow(df.iloc[i, :])
        writer_train_clf.writerow(df.iloc[i, :])

    else:
        writer_test.writerow(df.iloc[i, :])

# close the files:
f_train.close()
f_val.close()
f_test.close()
f_train_clf.close()

""""
# control:
df_test = pd.read_csv("new_all_images_test.csv")
test_size = len(df_test)
df_val = pd.read_csv("new_all_images_val.csv")
val_size = len(df_val)
df_train = pd.read_csv("new_all_images_train.csv")
train_size = len(df_train)
df_train_clf = pd.read_csv("new_all_images_train_clf.csv")
train_clf = len(df_train_clf)

original_size = len(df)

print("train_size", train_size)
print("val_size", val_size)
print("test_size", test_size)
print("They sum up to the initial number of images: ", original_size == test_size + val_size + train_size)
print("Classification train images composed of train and val: ", train_clf == val_size + train_size)
"""





