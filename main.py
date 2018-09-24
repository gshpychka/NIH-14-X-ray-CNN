import h5py
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import datetime as dt
import os

labels =


def load_labels_into_df():
    return pd.read_csv("data/labels.csv")