#%%

import h5py
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import datetime as dt
import os


#%%

def load_labels_into_df():
    return pd.read_csv("data/labels.csv")

#%%


labels = load_labels_into_df()
labels = labels[['Image Index', 'Finding Labels', 'Patient ID', 'Patient Age', 'Patient Gender']]

#%%

labels.head()

#%%

findings_list = [
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

# One-hot encoding
for finding in findings_list:
    labels[finding] = labels['Finding Labels'].apply(lambda x: 1 if finding in x else 0)


#%%

# Some disturbing outliers
labels['Patient Age'].sort_values(ascending=False).head(20)

#%%

plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((8,1), (0,0), rowspan=7)
ax2 = plt.subplot2grid((8,1), (7,0), rowspan=1)

data1 = pd.melt(labels,
             id_vars=['Patient Gender'],
             value_vars = list(findings_list),
             var_name = 'Category',
             value_name = 'Count')
data1 = data1.loc[data1.Count>0]

g=sns.countplot(y='Category',hue='Patient Gender',data=data1, ax=ax1, order = data1['Category'].value_counts().index)
ax1.set( ylabel="",xlabel="")
ax1.legend(fontsize=20)
ax1.set_title('Findings per gender',fontsize=18);

labels['No Finding']=labels['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)

data2 = pd.melt(labels,
             id_vars=['Patient Gender'],
             value_vars = list(['No Finding']),
             var_name = 'Category',
             value_name = 'Count')
data2 = data2.loc[data2.Count>0]
g=sns.countplot(y='Category',hue='Patient Gender',data=data2,ax=ax2)
ax2.set( ylabel="",xlabel="Count")
ax2.legend('')
plt.subplots_adjust(hspace=.5)

plt.show()

#%%

plt.figure(figsize=(15,10))
ax1 = plt.subplot()

counts = data1.groupby(['Category', 'Patient Gender']).count()

counts = counts.unstack(level=1)
counts.columns = counts.columns.droplevel(level=0)

#%%

percentages = counts.apply(lambda x: [x['M'] * 100 / x.sum(), 100 - (x['M'] * 100 / x.sum())], axis=1, result_type='broadcast')

#%%
percentages.plot(kind='barh', stacked=True, title='Findings distribution by gender')
plt.show()