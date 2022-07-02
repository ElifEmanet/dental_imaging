import pandas as pd
import numpy as np
import cv2

import os
import torch

import pydicom as dicom

from torch.utils.data import Dataset
from scipy import ndimage
from skimage.util import random_noise
from skimage.transform import resize


class OPGDataset(Dataset):
    """OPG images dataset."""

    def __init__(self, csv_file, img_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_path (string): Path to the folder where images are stored.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.frame = pd.read_csv(csv_file)
        self.img_path = img_path
        self.transform = transform

        # 2. column: the image paths/names
        self.image_arr = np.asarray(self.frame.iloc[:, 0])

        # 12. column: machine (0 or 1)
        self.machine_arr = np.asarray(self.frame.iloc[:, 10])

        # 13. column: classification_new (integer)
        self.cl_new_arr = np.asarray(self.frame.iloc[:, 11])

    def __len__(self):
        return len(self.frame)
        # return len(self.frame.index)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        single_image_name = self.image_arr[index] + '.dcm'

        image_path = os.path.join(self.img_path, single_image_name)

        # read dcm into an ndarray (! for tensors: switch axes??):
        image_dcm = dicom.dcmread(image_path)
        image = image_dcm.pixel_array.astype(np.uint)

        """
        # ToTensor method requires nparray of int [0, 255]
        # to rescale: 
        image = (np.maximum(image, 0)/ image_dcm.max())*255
        """

        # machine:
        machine = self.machine_arr[index]

        # cl_new:
        cl_new = self.cl_new_arr[index]

        # binary class:
        # artefacts = [8, 10, 11]
        artefacts = [8]
        if cl_new in artefacts:
            bin_class = 1
        else:
            bin_class = 0

        # sample:
        sample = {'id': self.image_arr[index],
                  'image': image,
                  'machine': machine,
                  'cl_new': cl_new,
                  'bin_class': bin_class}

        if self.transform:
            sample = self.transform(sample)

        return sample


class DataSubSet(Dataset):
    '''
    Dataset wrapper to apply transforms separately to subsets
    '''
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        sample = self.subset[i]

        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        if self.transform:
            augmented = self.transform(sample)
            id = augmented['id']
            image = augmented['image']
            machine = augmented['machine']
            cl_new = augmented['cl_new']
            bin_class = augmented['bin_class']

        return {'id': id, 'image': image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class AdjustContrast(object):
    def __init__(self, s, lmda, epsilon):
        assert isinstance(s, float)  # Goodfellow et al. (2013a) used s = 1.0
        assert isinstance(lmda, float)  # Coates et al. (2011) used lambda = 10.0
        assert isinstance(epsilon, float)  # Coates et al. (2011) used epsilon = 0.0

        self.s = s
        self.lmda = lmda
        self.epsilon = epsilon

    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        mean = image.mean()
        image = image - mean
        contrast = np.sqrt(self.lmda + np.mean(image ** 2))
        image = self.s * image / max(contrast, self.epsilon)
        # image = np.expand_dims(image, axis=-1)

        return {'id': id, 'image': image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class NormalizeIntensity(object):
    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        max = image.max()
        image = image/max
        # image = np.expand_dims(image, axis=-1)

        return {'id': id, 'image': image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class Center(object):
    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        mean = image.mean()
        image = image - mean
        # image = np.expand_dims(image, axis=-1)

        return {'id': id, 'image': image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class Rotate(object):
    def __init__(self, angle, reshape, mode):
        assert isinstance(angle, float)
        assert isinstance(reshape, bool)
        assert isinstance(mode, str)

        self.angle = angle
        # If you want the output shape to be adapted s.t. the input array is contained completely in the output: set reshape to True (default)
        self.reshape = reshape
        self.mode = mode

    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        img_rotated = ndimage.rotate(image, self.angle, reshape=self.reshape, mode=self.mode)
        # img_rotated = np.expand_dims(img_rotated, axis=-1)

        return {'id': id, 'image': img_rotated, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class RandomNoise(object):
    def __init__(self, mode, ratio):
        assert isinstance(ratio, float)
        assert isinstance(mode, str)

        self.ratio = ratio
        self.mode = mode

    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        noise_img = random_noise(image, mode=self.mode)
        # noise_img = np.expand_dims(noise_img, axis=-1)

        return {'id': id, 'image': noise_img, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class RandomCropAndResize(object):
    def __init__(self, crop_height, crop_width, width, height, mode):
        assert isinstance(crop_height, int)
        assert isinstance(crop_width, int)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(mode, str)

        self.size = (width, height)
        self.width = width
        self.height = height

        self.mode = mode

        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        max_x = image.shape[1] - self.crop_width
        max_y = image.shape[0] - self.crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = image[y: y + self.crop_height, x: x + self.crop_width]
        resized_image = resize(crop, (self.width, self.height), mode=self.mode)
        # resized_image = np.expand_dims(resized_image, axis=-1)

        return {'id': id, 'image': resized_image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class Sharpen(object):
    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        # kernel to sharpen the image
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        # ddepth=-1 means that the output image will have the same depth as the input image
        image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        image_sharp = np.expand_dims(image_sharp, axis=-1)

        return {'id': id, 'image': image_sharp, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class Blur(object):
    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        blurred_image = cv2.blur(image, (7, 7))
        # blurred_image = np.expand_dims(blurred_image, axis=-1)

        return {'id': id, 'image': blurred_image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class Resize(object):
    def __init__(self, width, height, mode):
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(mode, str)

        self.size = (width, height)
        self.mode = mode

    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        resized_image = resize(image, self.size, mode=self.mode)
        # resized_image = np.expand_dims(resized_image, axis=-1)

        return {'id': id, 'image': resized_image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class Zoom(object):
    def __init__(self, zoom_factor):
        assert isinstance(zoom_factor, float)

        self.zoom_factor = zoom_factor

    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        if self.zoom_factor == 0:
            return {'id': id, 'image': image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}

        else:
            y_size = image.shape[0]
            x_size = image.shape[1]

            # define new boundaries
            x1 = int(0.5 * x_size * (1 - 1 / self.zoom_factor))
            x2 = int(x_size - x1)
            y1 = int(0.5 * y_size * (1 - 1 / self.zoom_factor))
            y2 = int(y_size - y1)

            # first crop image then scale
            img_cropped = image[y1:y2, x1:x2]
            zoomed_image = cv2.resize(img_cropped, None, fx=self.zoom_factor, fy=self.zoom_factor)

            return {'id': id, 'image': zoomed_image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class RescalePixelDims(object):
    def __init__(self, machine_0_dim, machine_1_dim):
        assert isinstance(machine_0_dim, float)
        assert isinstance(machine_1_dim, float)

        self.middle = (machine_0_dim + machine_1_dim) / 2
        self.rescale_factor_0 = self.middle / machine_0_dim
        self.rescale_factor_1 = self.middle / machine_1_dim

    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        if machine == 0:
            rescaled_image = cv2.resize(image, None, fx=self.rescale_factor_0, fy=self.rescale_factor_0)

        else:
            rescaled_image = cv2.resize(image, None, fx=self.rescale_factor_1, fy=self.rescale_factor_1)

        # resized_image = np.expand_dims(resized_image, axis=-1)

        return {'id': id, 'image': rescaled_image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class ExpandDims(object):
    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        image = np.expand_dims(image, axis=-1)

        return {'id': id, 'image': image, 'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        id = sample['id']
        image = sample['image']
        machine = sample['machine']
        cl_new = sample['cl_new']
        bin_class = sample['bin_class']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))

        return {'id': id, 'image': torch.from_numpy(image).type(torch.DoubleTensor),
                'machine': machine, 'cl_new': cl_new, 'bin_class': bin_class}
