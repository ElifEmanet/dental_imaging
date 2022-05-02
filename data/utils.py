import SimpleITK as sitk
import numpy as np
import os
import cv2


def load_data(dir_images):
    image_ids = []
    images_npa = []
    
    for file in os.listdir(dir_images):

        # load images
        path_image = os.path.join(dir_images, file)
        image = sitk.ReadImage(path_image)

        # save img id
        image_ids.append(file[:-4])

        # get numpy array from image
        image_npa = sitk.GetArrayFromImage(image)[0,:,:]
        images_npa.append(image_npa)

    return image_ids, images_npa

def resize(dataset, size):
    reduced_set = []
    # resize images
    for i in range(len(dataset)):
        reduced_set.append(cv2.resize(dataset[i], dsize=size, interpolation=cv2.INTER_CUBIC))
    # add dimension
    reduced_set = np.expand_dims(reduced_set, axis=-1)
    return reduced_set

def max_normalize(dataset):
    norm_data = dataset.astype("float32") / np.max(dataset)
    return norm_data

def contrast_normalize(dataset, s, lmda, epsilon):
    normalized_set = []
    for i in range(len(dataset)):
        X = dataset[i]
        X_average = np.mean(dataset[i])
        X = X - X_average
        contrast = np.sqrt(lmda + np.mean(X**2))
        X = s * X / max(contrast, epsilon)
        normalized_set.append(X)
    return normalized_set