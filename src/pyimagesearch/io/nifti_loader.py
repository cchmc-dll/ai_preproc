# import the necessary packages
import os
import glob
import tables
import numpy as np
import pandas as pd
import nibabel as nib
from ...unet3d.normalize import reslice_image_set, get_volume


class nifti_loader:
    def __init__(self, hdf5, input_images, problem_type='Classification', input_shape=(128, 128, 32), image_modalities=['T2'], mask=None):
        self.input_images = input_images
        self.input_shape = input_shape
        self.problem_type = problem_type
        self.image_modalities = image_modalities
        self.mask = mask
        self.hdf5 = hdf5
        if self.problem_type == 'Segmentation':
            training_data_files = list()
            subject_ids = list()
            for subject_dir in glob.glob(os.path.join(self.input_images, "*")):
                subject_ids.append(os.path.basename(subject_dir))
                subject_files = list()
                for modality in self.image_modalities + self.mask:
                    subject_files.append(os.path.join(
                        subject_dir, modality + ".nii"))
                training_data_files.append(tuple(subject_files))
            self.data_files = training_data_files
            self.ids = subject_ids
            self.image_data_shape = tuple(
                [0, len(self.image_modalities)] + list(self.input_shape))
            self.truth_data_shape = tuple([0, 1] + list(self.input_shape))
            self.n_channels = len(self.image_modalities)
        elif self.problem_type == 'Classification':
            training_data_files = list()
            subject_ids = list()
            for classes in glob.glob(os.path.join(self.input_images, "*")):
                for subject_dir in glob.glob(os.path.join(classes, "*")):
                    subject_ids.append(os.path.basename(subject_dir))
                    subject_files = list()
                    for modality in self.image_modalities + self.mask:
                        subject_files.append(os.path.join(
                            subject_dir, modality + ".nii"))
                    training_data_files.append(tuple(subject_files))
            self.data_files = training_data_files
            self.ids = subject_ids
            self.image_data_shape = tuple(
                [0, len(self.image_modalities)+len(self.mask)] + list(self.input_shape))
            self.truth_data_shape = tuple([0, ])
            self.n_channels = len(self.image_modalities)+len(self.mask)

        # elif problem_type is 'Regression':
        #    training_data_files = list()
        #    subject_ids = list()
        #    for subject_dir in glob.glob(os.path.join(self.input_images, "*")):
        #        subject_ids.append(os.path.basename(subject_dir))
        #        subject_files = list()
        #        for modality in self.image_modalities:
        #            subject_files.append(os.path.join(subject_dir, modality + ".nii"))
        #        training_data_files.append(tuple(subject_files))
        #    self.data_files = training_data_files
        #    self.ids = subject_ids

    def get_sample_ids(self):
        return self.ids

    def set_sample_ids(self, new_ids):
        self.ids = new_ids

    def load_toHDF5(self, verbose=-1):
                    # initialize the list of features and labels
        n_samples = len(self.ids)
        filters = tables.Filters(complevel=5, complib='blosc')
        image_storage = self.hdf5.create_earray(self.hdf5.root, 'imdata', tables.Float32Atom(
        ), shape=self.image_data_shape, filters=filters, expectedrows=n_samples)
        affine_storage = self.hdf5.create_earray(self.hdf5.root, 'affine', tables.Float32Atom(
        ), shape=(0, 4, 4), filters=filters, expectedrows=n_samples)
        # Stores image stats: ["input image width","input image height","input image depth","input pixel width","input pixel height","input slice thickness",
        #                      "output image width","output image height","output image depth","output pixel width","output pixel height","output slice thickness"]
        imstats_storage = self.hdf5.create_earray(self.hdf5.root, 'imstats', tables.Float32Atom(
        ), shape=(0, 12), filters=filters, expectedrows=n_samples)
        if self.problem_type == "Classification":
            truth_storage = self.hdf5.create_earray(self.hdf5.root, 'truth', tables.StringAtom(
                itemsize=15), shape=self.truth_data_shape, filters=filters, expectedrows=n_samples)
        elif self.problem_type == "Segmentation":
            truth_storage = self.hdf5.create_earray(self.hdf5.root, 'truth', tables.UInt8Atom(
            ), shape=self.truth_data_shape, filters=filters, expectedrows=n_samples)
            # Separate vector of volume of the Segmentation mask
            volume_storage = self.hdf5.create_earray(self.hdf5.root, 'volume', tables.Float32Atom(
            ), shape=(0, 1), filters=filters, expectedrows=n_samples)

        # loop over the input images
        for (i, imagePath) in enumerate(self.data_files):
                        # load the image and extract the class label assuming
                        # that our path has the following format:
                        # /path/to/dataset/{class}/{image}.jpg
            if self.problem_type == "Classification":
                subject_name = imagePath[0].split(os.path.sep)[-2]
                if subject_name in self.ids:
                    images, imstats = reslice_image_set(
                        in_files=imagePath, image_shape=self.input_shape, label_indices=len(imagePath)-1, crop=True, stats=True)
                    label = imagePath[0].split(os.path.sep)[-3]
                    subject_data = [image.get_data() for image in images]
                    affine = images[0].affine

                    image_storage.append(np.asarray(subject_data)[np.newaxis])
                    imstats_storage.append(np.asarray(imstats)[np.newaxis])
                    affine_storage.append(np.asarray(affine)[np.newaxis])
                    truth_storage.append(np.asarray(label)[np.newaxis])

            elif self.problem_type == "Segmentation":
                images, imstats = reslice_image_set(
                    in_files=imagePath, image_shape=self.input_shape, label_indices=len(imagePath)-1, crop=True, stats=True)
                subject_data = [image.get_data() for image in images]
                affine = images[0].affine
                volume = get_volume(imstats[6:], mask=np.asarray(
                    subject_data[self.n_channels]), label=1, units="cm")
                image_storage.append(np.asarray(
                    subject_data[:self.n_channels])[np.newaxis])
                imstats_storage.append(np.asarray(imstats)[np.newaxis])
                affine_storage.append(np.asarray(affine)[np.newaxis])
                volume_storage.append(np.asarray(volume)[np.newaxis][np.newaxis])
                truth_storage.append(np.asarray(subject_data[self.n_channels], dtype=np.uint8)[
                                     np.newaxis][np.newaxis])

            # elif self.problem_type is "Regression":
            #    image = cv2.imread(imagePath)

                # show an update every `verbose` images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(self.ids)))
        return(image_storage)

    def hdf5_toImages(self, output_dir):
        for index in range(0,len(self.hdf5.root.imdata)):
            if 'subject_ids' in self.hdf5.root:
                case_directory = os.path.join(
                    output_dir, self.hdf5.root.subject_ids[index].decode('utf-8'))
            else:
                case_directory = os.path.join(
                    output_dir, "validation_case_{}".format(index))

            if not os.path.exists(case_directory):
                os.makedirs(case_directory)

            affine = self.hdf5.root.affine[index]
            test_data = np.asarray([self.hdf5.root.imdata[index]])
            for i, modality in enumerate(self.image_modalities):
                image = nib.Nifti1Image(test_data[0, i], affine)
                # Set the xyzt units header to 2 (mm)
                image.header['xyzt_units'] = 2
                image.to_filename(os.path.join(
                    case_directory, "data_{0}.nii.gz".format(modality)))

            test_truth = nib.Nifti1Image(
                self.hdf5.root.truth[index][0], affine)
            # Set the xyzt units header to 2 (mm)
            test_truth.header['xyzt_units'] = 2
            test_truth.to_filename(os.path.join(
                case_directory, "truth.nii.gz"))

    def hdf5_toImStats(self, output_dir, output_file="imstats.csv"):
        if self.hdf5:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            subject_ids = np.asarray(self.hdf5.root.subject_ids)[:,np.newaxis].astype('U13')
            imstats = np.asarray(self.hdf5.root.imstats).astype('U13')
            volume = np.asarray(self.hdf5.root.volume).astype('U13')
            header = ['ID', 'inputX', 'inputY', 'inputZ', 'inputXsize', 'inputYsize', 'inputZsize',
                    'outputX', 'outputY', 'outputZ', 'outputXsize', 'outputYsize', 'outputZsize', 'volume']
            fdata = np.concatenate((subject_ids, imstats, volume), axis=1)
            df = pd.DataFrame(data=fdata, columns=header)
            df.to_csv(os.path.join(output_dir, output_file), encoding='utf-8')
