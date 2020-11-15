import os
import glob
import cmd
import sys
from argparse import ArgumentParser
from .pyimagesearch.io.nifti_loader import nifti_loader
from .pyimagesearch.io.TIF_loader import TIF_loader, MiddleTIFLoader
import numpy as np
import pandas as pd
import tables
import pprint
from .unet3d.normalize import normalize_data_storage, normalize_clinical_storage, normalize_data_storage_2D
from .unet3d.generator import get_validation_split

config = dict()

def parse_command_line_arguments():
    print('argv type: ', type(sys.argv))
    print('argv: ', sys.argv)
    parser = ArgumentParser(fromfile_prefix_chars='@')

    req_group = parser.add_argument_group(title='Required flags')
    req_group.add_argument(
        '--problem_type',
        required=True,
        help='Segmentation, Classification, or Regression, default=Segmentation')
    req_group.add_argument(
        '--output_file',
        required=True,
        help='Enter the name of the .h5 file to be saved after preprocessing input data')
    req_group.add_argument(
        '--image_masks',
        required=True,
        help='Comma separated list of mask names, ex: Muscle,Bone,Liver')
    req_group.add_argument(
        '-o,',
        '--output_dir',
        required=True,
        help='Path to directory where output files will be saved')
    req_group.add_argument(
        '--all_modalities',
        required=True,
        help='Comma separated list of desired image modalities')
    req_group.add_argument(
        '--image_format',
        required=True,
        help='TIF or NIFTI')
    parser.add_argument(
        '--input_images',
        help='Folder containing input images')
    parser.add_argument(
        '--CPU',
        default=4,
        type=int,
        help='Number of CPU cores to use, default=4')
    parser.add_argument('--patch_shape', default=None)
    parser.add_argument(
        '--input_type',
        default='Image',
        help='Image,Clinical,Both')
    parser.add_argument('--image_shape', default='256,256')
    parser.add_argument(
        '--overwrite',
        default=1,
        type=int,
        help='0=false, 1=true')
    parser.add_argument(
        '--output_images',
        default=0,
        type=int,
        help='1=True,0=False, To write preprocessed output as individual images in addition to h5 file')
    parser.add_argument(
        '--output_imstats',
        default=0,
        type=int,
        help='1=True,0=False, To write preprocessed image statistics to a csv file')
    parser.add_argument(
        '--training_modalities',
        help='Comma separated list of desired image modalities for training only')
    parser.add_argument(
        '--normalize',
        default=1,
        type=int,
        help='0=false, 1=true, Flag to Normalize images after resizing ')
    return parser.parse_args()

def build_config_dict(config):
   # config["labels"] = tuple(config['labels'].split(','))  # the label numbers on the input image
   # config["n_labels"] = len(config["labels"])

    config['all_modalities'] = config['all_modalities'].split(',')

    try:
        config["training_modalities"] = config['training_modalities'].split(
            ',')  # change this if you want to only use some of the modalities
    except AttributeError:
        config["training_modalities"] = config['all_modalities']

    # calculated values from cmdline_args
    config["n_channels"] = len(config["training_modalities"])
    mapper = lambda x: None if x == 'None' else int(x)
    config["image_shape"] = map(mapper,config['image_shape'].split(','))
    config["image_shape"] = tuple(list(config["image_shape"]))
    config["image_masks"] = config['image_masks'].split(',')
    config["output_file"] = os.path.join(config["output_dir"], config["output_file"])

    # Save absolute path for input folders
    if (config["input_type"] == "Image" or config["input_type"] == "Both"):
        try:
            config["input_images"] = os.path.abspath(config["input_images"])
        except BaseException:
            print(
                "Error: Input Image Folder for preprocessing not defined | \t Set config[\"input_images\"] \n")

    if (config["input_type"] == "Clinical" or config["input_type"] == "Both"):
        try:
            config["input_clinical"] = os.path.abspath(
                config["input_clinical"])
        except BaseException:
            print(
                "Error: Input Clinical Folder with .csv for preprocessing not defined | \t Set config[\"input_clinical\"] \n")

    return config

def main(*arg):
    if arg:
        sys.argv.append(arg[0])
    args = parse_command_line_arguments()
    pprint.pprint(args)
    config = build_config_dict(vars(args))
    pprint.pprint(config)
    #run_preprocess(config)

    return config
    # # Open the hdf5 file
    # if config['overwrite'] or not os.path.exists(config["output_file"]):
    #     hdf5_file = tables.open_file(config["output_file"], mode='w')
    #     overwrite = 1
    # else:
    #     hdf5_file = tables.open_file(config["output_file"], mode='r')

    # config["hdf5_file"] = hdf5_file

    # niftis = nifti_loader(
    #         config["hdf5_file"],
    #         config["input_images"],
    #         config["problem_type"],
    #         config["image_shape"],
    #         config["training_modalities"],
    #         config["image_masks"]
    #     )

    # print(' Image data shape: ', niftis.image_data_shape)
    # print(' Image modalities: ', niftis.image_modalities)
    # print('  No. of Subjects: ', len(niftis.ids))
    # niftis.load_toHDF5()

if __name__ == "__main__":
    main()
