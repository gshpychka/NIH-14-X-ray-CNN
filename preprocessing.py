#%%
import cv2
import h5py
import datetime as dt
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_findings_list():
    return [
        'Cardiomegaly',
        'Emphysema',
        'Effusion',
        'Hernia',
        'Nodule',
        'Pneumothorax',
        'Atelectasis',
        'Pleural_Thickening',
        'Mass',
        'Edema',
        'Consolidation',
        'Infiltration',
        'Fibrosis',
        'Pneumonia']

def load_labels_into_df():
    return pd.read_csv("data/labels.csv")

def preprocess_labels():
    age_range = range(18, 95)

    labels = load_labels_into_df()

    # Only non-adults and weird values
    labels = labels[labels['Patient Age'].isin(age_range)]
    # Keep only the relevant columns
    labels = labels[['Image Index', 'Finding Labels', 'Patient ID', 'Patient Age', 'Patient Gender']]

    findings_list = get_findings_list()

    # One-hot encoding
    for finding in findings_list:
        labels[finding] = labels['Finding Labels'].apply(lambda x: 1 if finding in x else 0)
    labels['No Finding'] = labels['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)
    labels['Any Finding'] = labels['Finding Labels'].apply(lambda x: 0 if 'No Finding' in x else 1)
    return labels

labels = preprocess_labels()

def store_labels_hdf5(file = None):
    if file is None:
        file = h5py.File(os.path.abspath('data/data_new.h5'), 'w')

    label_group = file.create_group('labels')
    # storing numerical labels
    numerical_labels = ['Patient ID', 'Patient Age', 'No Finding', 'Any Finding'] + get_findings_list()
    for label in numerical_labels:
        label_group.create_dataset(
            name=label,
            data=labels[label],
            shape=labels[label].shape,
        )
    # storing string labels
    string_labels = ['Image Index', 'Patient Gender']
    for label in string_labels:
        string_dt = h5py.special_dtype(vlen=str)
        data = np.asarray(labels[label], dtype=object)
        label_group.create_dataset(label, data=data, dtype=string_dt)

    file.close()


def conv_multiple_dsets_to_one(file_old=None, file_new=None):
    if file_old is None:
        file_old = h5py.File(os.path.abspath('data/data.h5'), 'r')

    if file_new is None:
        file_new = h5py.File(os.path.abspath('data/data_new.h5'), 'w')


    chunk = 1000;
    start = dt.datetime.now()

    Xset = file_new.create_dataset(
        name='X',
        data=None,
        shape=(len(labels),224,224),
        maxshape=(None, 224, 224),
        chunks=None,
        compression='gzip',
        compression_opts=9
    )

    for k in range(0, int(len(labels) / chunk)):
        X = np.empty((chunk, 224, 224))
        for i in range(0, chunk):
            img_num = chunk * k + i
            X[i] = file_old[labels['Image Index'][i]][:]
            print("Transferred image %d into array." % img_num)
        Xset[k*chunk:] = X
    end = dt.datetime.now()
    print("Done, in %s, yo." % (end - start))
    file_old.close()
    file_new.close()


def preprocess_images_into_separate_datasets(file=None):
    if file is None:
        file =  h5py.File(os.path.abspath('data/data.h5'), 'w')

    start = dt.datetime.now()
    PATH = os.path.abspath('../NIHChestXrayDataset')

    # filenames
    images = labels['Image Index']
    print('There are %d images in the dataset.' % len(images))

    width = 224
    height = width

    for i, img in enumerate(images):
        # images
        image = cv2.imread(os.path.join(PATH, img))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        Xset = file.create_dataset(
            name=img,
            data=image,
            shape=(width, height),
            maxshape=(width, height),
            compression='gzip',
            compression_opts=9
        )
        print('Processed image %d' % i)
    end = dt.datetime.now()
    print('Processed %d images in %s' % (i, (end - start)))

    file.close()




def test(filename='data.h5', num=5):
    file = h5py.File(filename, 'r')
    labels = file.require_group('labels')
    image_index = labels['Image Index'][num]
    image = np.array(file[image_index])

    print("Image shape: " + str(image.shape))
    cv2.imwrite(str(labels['Image Index'][num]), image)
    print("%s patiend ID %d, %d years old, has %s" % (
        labels['Patient Gender'][num],
        labels['Patient ID'][num],
        labels['Patient Age'][num],
        labels['Finding Labels'][num]))
    print('Saved image downscaled from ' + labels['Image Index'][num])
    os.system('cp data/%s %s-original.png' % (
        image_index,
        image_index))