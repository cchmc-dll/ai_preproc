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
    config["output_file"] = os.path.join(
        config["output_dir"], config["output_file"])

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


def get_image_loader(config):
    if config["image_format"] == "TIF":
        if config['use_middle_image']:
            return MiddleTIFLoader(
                config["problem_type"],
                config["input_images"],
                config["image_shape"],
                config["training_modalities"],
                config["image_masks"],
                config['slice_number']
            )
        else:
            return TIF_loader(
                config["problem_type"],
                config["input_images"],
                config["image_shape"],
                config["training_modalities"],
                config["image_masks"],
                config['slice_number']
            )
    elif config["image_format"] == 'NIFTI':
        return nifti_loader(
            config["hdf5_file"],
            config["input_images"],
            config["problem_type"],
            config["image_shape"],
            config["training_modalities"],
            config["image_masks"]
        )
    else:
        raise RuntimeError(
            f'Unsupported image format: {config["image_format"]}')


def main(*arg):
    if arg:
        sys.argv.append(arg[0])
    args = parse_command_line_arguments()
    config = build_config_dict(vars(args))
    pprint.pprint(config)
    run_preprocess(config)


def run_preprocess(config):

    # Step 3: Check if Output file already exists, If it exists, require user
    # permission to overwrite
    if 'overwrite' in config:
        overwrite = config["overwrite"]
    elif os.path.exists(config["output_file"]):
        overwrite = input(
            "Output file exists, do you want to overwrite? (y/n) \n")
        overwrite = True if overwrite == 'y' else False

    # Open the hdf5 file
    if overwrite or not os.path.exists(config["output_file"]):
        hdf5_file = tables.open_file(config["output_file"], mode='w')
        overwrite = 1
    else:
        hdf5_file = tables.open_file(config["output_file"], mode='r')

    config["hdf5_file"] = hdf5_file
    # Step 4: Check problem specific parameters are defined
    if (config["input_type"] == "Both"):
        # Step 6: Load Imaging Data to hdf5 after checking if samples have both
        # image and clinical data. If any 1 is missing, those samples are
        # neglected.
        image_loader = get_image_loader(config)
        subject_ids = image_loader.get_sample_ids()
        image_storage = None
        id_storage = None
        df_features = pd.read_csv(
            os.path.join(
                config["input_clinical"],
                'Features.csv'))
        df_features.set_index('Key', inplace=True)
        # If Both, select only samples that have both clinical and imaging data
        features = list(df_features)
        feature_array = []
        subject_ids_final = []
        for i, subject in enumerate(subject_ids):
            if subject in df_features.index:
                feature_array.append(df_features.loc[subject, features])
                subject_ids_final.append(subject)
        image_loader.set_sample_ids(subject_ids_final)
        image_storage = image_loader.load_toHDF5(hdf5_file)

        # Load Clinical data to hdf5
        feature_array = np.asarray(feature_array)
        clinical_storage = hdf5_file.create_array(
            hdf5_file.root, 'cldata', obj=feature_array)
        id_storage = hdf5_file.create_array(
            hdf5_file.root, 'subject_ids', obj=subject_ids_final)
        print("Input Data Preprocessed and Loaded to HDF5")

        # Step 7: Normalize Data Storage
        if config["normalize"]:
            normalize_data_storage(image_storage)
            normalize_clinical_storage(clinical_storage)
            print("Data in HDF5 File is normalized for training")

    elif (config["input_type"] == "Image"):
        # Step 6: Load Imaging Data
        image_loader = get_image_loader(config)
        if overwrite:
            image_storage = image_loader.load_toHDF5()
            subject_ids_final = image_loader.get_sample_ids()
            id_storage = hdf5_file.create_array(
                hdf5_file.root, 'subject_ids', obj=subject_ids_final)

            print("Input Data Preprocessed and Loaded to HDF5")

            # Step 7: Normalize Data Storage
            if config["normalize"]:
                if len(config["image_shape"]) > 2:
                    normalize_data_storage(image_storage)
                    print("Data in HDF5 File is normalized for training")
                else:
                    normalize_data_storage_2D(image_storage)

        if(config['output_images']):
            image_loader.hdf5_toImages(output_dir=config['output_file'].split('/')[0])

        if(config['output_imstats']):
            image_loader.hdf5_toImStats(output_dir=config['output_file'].split('/')[0])

    # Step 6: Load Clinical data
    elif (config["input_type"] == "Clinical"):
        df_features = pd.read_csv(
            os.path.join(
                config["input_clinical"],
                'Features.csv'))
        df_features.set_index('Key', inplace=True)
        # If Both, select only samples that have both clinical and imaging data
        features = list(df_features)
        feature_array = []
        subject_ids = []
        truth_storage = None
        subject_ids = df_features.index
        feature_array = df_features[features]

        df_truth = pd.read_csv(
            os.path.join(
                config["input_clinical"],
                'Truth.csv'))
        df_truth.set_index('Key', inplace=True)
        truth = df_truth.loc[subject_ids, config["clinical_truthname"]]
        truth = truth.tolist()
        subject_ids = subject_ids.tolist()
        feature_array = np.array(feature_array)
        truth = np.asarray(truth)

        clinical_storage = hdf5_file.create_array(
            hdf5_file.root, 'cldata', obj=feature_array)
        truth_storage = hdf5_file.create_array(
            hdf5_file.root, 'truth', obj=truth)
        id_storage = hdf5_file.create_array(
            hdf5_file.root, 'subject_ids', obj=subject_ids)

        # Step 7: Normalize Data Storage
        if config["normalize"]:
            normalize_clinical_storage(clinical_storage)
            print("Data in HDF5 File is normalized for training")

    hdf5_file.close()


if __name__ == "__main__":
    main()
