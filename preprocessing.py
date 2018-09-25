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

def load_labels_into_df(shuffle=False):
    labels = pd.read_csv("data/labels.csv")
    if shuffle:
        return labels.sample(frac=1).reset_index(drop=True)
    else:
        return labels

def preprocess_labels(shuffle=False):
    age_range = range(18, 95)

    labels_df = load_labels_into_df(shuffle)

    # Drop non-adults and weird values
    labels_df = labels_df[labels_df['Patient Age'].isin(age_range)]
    # Keep only the relevant columns
    labels_df = labels_df[['Image Index', 'Finding Labels', 'Patient ID', 'Patient Age', 'Patient Gender']]

    findings_list = get_findings_list()

    # One-hot encoding
    for finding in findings_list:
        labels_df[finding] = labels_df['Finding Labels'].apply(lambda x: 1 if finding in x else 0)
    labels_df['No Finding'] = labels_df['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)
    labels_df['Any Finding'] = labels_df['Finding Labels'].apply(lambda x: 0 if 'No Finding' in x else 1)

    labels_df = labels_df.reset_index(drop=True)
    return labels_df

labels = preprocess_labels()


def store_labels_hdf5(labels=labels, file = None, mode='a'):
    if file is None:
        file = h5py.File(os.path.abspath('data/data_new.h5'), mode)
    label_group = file.require_group('labels')
    # storing numerical labels
    numerical_labels = ['Patient ID', 'Patient Age', 'No Finding', 'Any Finding'] + get_findings_list()
    for label in numerical_labels:
        label_group.create_dataset(
            name=label,
            data=labels[label],
            shape=labels[label].shape,
        )
    # storing string labels
    string_labels = ['Image Index', 'Finding Labels', 'Patient Gender']
    for label in string_labels:
        string_dt = h5py.special_dtype(vlen=str)
        data = np.asarray(labels[label], dtype=object)
        label_group.create_dataset(label, data=data, dtype=string_dt)
    file.close()


def conv_multiple_dsets_to_one(file_old=None, file_new=None, mode='a', limit=float('inf'), chunk=2000, dset_chunks=(1, 224, 224, 1), compression='gzip'):
    start = dt.datetime.now()
    if file_old is None:
        file_old = h5py.File(os.path.abspath('data/data.h5'), 'r')

    if file_new is None:
        file_new = h5py.File(os.path.abspath('data/data_new.h5'), mode)

    Xset = file_new.require_dataset(
        name='X',
        data=None,
        shape=(len(file_new['labels']['Image Index']),224,224, 1),
        maxshape=(len(file_new['labels']['Image Index']), 224, 224, 1),
        chunks=dset_chunks,
        compression=compression,
        compression_opts=9,
        dtype='<f4'
    )

    for k in range(0, int(len(file_new['labels']['Image Index']) / chunk)):
        if k * chunk > limit:
            break
        X = np.empty((chunk, 224, 224))
        for i in range(0, chunk):
            img_num = chunk * k + i
            if img_num > len(file_new['labels']['Image Index']) or img_num > limit:
                break
            X[i] = file_old[file_new['labels']['Image Index'][img_num]][:]
            print("Transferred image %d into array." % img_num)
        print("Recording to indices %d to %d" % ((k*chunk), (k+1)*chunk))
        Xset[k*chunk:(k+1)*chunk] = X[..., np.newaxis]
    end = dt.datetime.now()
    print("Done, in %s, yo." % (end - start))
    file_old.close()
    file_new.close()


def preprocess_images_into_separate_datasets(file=None):
    if file is None:
        file =  h5py.File(os.path.abspath('data/data.h5'), 'a')

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




def test(filename='data/data_new.h5', num=5):
    file = h5py.File(filename, 'r')
    labels = file.require_group('labels')
    image_index = labels['Image Index'][num]
    image = np.array(file['X'][num])

    print("Image shape: " + str(image.shape))
    cv2.imwrite(str(labels['Image Index'][num]), image)
    print("%s patiend ID %d, %d years old, has %s" % (
        labels['Patient Gender'][num],
        labels['Patient ID'][num],
        labels['Patient Age'][num],
        labels['Finding Labels'][num]))
    print('Saved image downscaled from ' + labels['Image Index'][num])
    os.system('cp ../NIHChestXrayDataset/%s %s-original.png' % (
        image_index,
        image_index))
    file.close()