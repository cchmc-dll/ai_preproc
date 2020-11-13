import pickle
import os
import collections

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like, resample_img

from .nilearn_custom_utils.nilearn_utils import crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """

    :param image_files:
    :param image_shape:
    :param crop:
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :return:
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "linear"
        image_list.append(read_image(
            image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return image_list

def read_image_files_wStats(image_files, image_shape=None, crop=None, label_indices=None):
    """

    :param image_files:
    :param image_shape:
    :param crop:
    :param use_nearest_for_last_file: If True, will use nearest neighbor interpolation for the last file. This is used
    because the last file may be the labels file. Using linear interpolation here would mess up the labels.
    :imstats - outputs the original resolution of the image and the final resolution for storage
    :return: images,imstats
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    imstats_list =[]
    for index, image_file in enumerate(image_files):
        if (label_indices is None and (index + 1) == len(image_files)) \
                or (label_indices is not None and index in label_indices):
            interpolation = "nearest"
        else:
            interpolation = "continuous"
        im = read_resize_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation)
        image_list.append(im)
        # Calculate image stats only for the first image (which would be the data file)
        if(index == 0):
            imstats_list = get_imstats(image_file) + get_imstats(im)

    return image_list,imstats_list


def read_resize_image(in_file, image_shape=None, interpolation='linear', crop=None):
    """
    Resizes nifti based on nibabel functions instead of SITK library used by read_image()
    :returns - resized image with proper affine matrices
    """
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        new_shape = image_shape
        # Reorder (get rids of rotations in the affine)
        image = reorder_img(image,resample=interpolation)
        # Calculate voxel spacing for desired image shape
        zoom_level = np.divide(new_shape, image.shape)
        current_spacing = [1, 1, 1]*image.affine[ :3, :3]
        new_spacing = np.divide(current_spacing, zoom_level)
        # Calculate the new affine matrix
        new_affine = np.eye(4)
        new_affine[ :3, :3] = np.eye(3)*new_spacing
        new_affine[ :3, 3] = image.affine[ :3, 3]
        # Resample new image
        image = resample_img(image, target_shape=new_shape, target_affine=new_affine, interpolation=interpolation)

    return image




def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3,3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)


def get_imstats(input):
    """
    :param input: either nifti image or image file
    :return: imstats: [X dim,Y dim,Z dim,Pixel size,Slice Thickness]
    """
    if isinstance(input, str):
        im = nib.load(os.path.abspath(input))
    else:
        im = input

    imshape = list(im.shape)
    voxsizes = im.header['pixdim'][1:4]
    return (imshape+voxsizes.tolist())




