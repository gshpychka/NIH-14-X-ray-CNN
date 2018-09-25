#%%

import h5py
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
sns.set()
import datetime as dt
import os
import preprocessing

#%%
labels = preprocessing.preprocess_labels()
findings_list = preprocessing.get_findings_list()

#%%

plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((8,1), (0,0), rowspan=7)
ax2 = plt.subplot2grid((8,1), (7,0), rowspan=1)

data1 = pd.melt(labels,
             id_vars=['Patient Gender'],
             value_vars=findings_list,
             var_name='Finding',
             value_name='Count')
data1 = data1.loc[data1.Count>0]

g=sns.countplot(y='Finding',hue='Patient Gender',data=data1, ax=ax1, order = data1['Finding'].value_counts().index)
ax1.set( ylabel="",xlabel="")
ax1.legend(fontsize=20)
ax1.set_title('Findings per gender',fontsize=18);

data2 = pd.melt(labels,
             id_vars=['Patient Gender'],
             value_vars=['No Finding'],
             var_name='Finding',
             value_name='Count')
data2 = data2.loc[data2.Count>0]
g=sns.countplot(y='Finding',hue='Patient Gender',data=data2,ax=ax2)
ax2.set( ylabel="",xlabel="Count")
ax2.legend('')
plt.subplots_adjust(hspace=.5)

plt.show()

#%%
# Gender ditributions per finding
plt.figure(figsize=(15,10))
ax1 = plt.subplot()
counts = data1.groupby(['Finding', 'Patient Gender']).count()
counts = counts.unstack(level=1)
counts.columns = counts.columns.droplevel(level=0)
percentages = counts.apply(lambda x: [x['M'] * 100 / x.sum(), 100 - (x['M'] * 100 / x.sum())], axis=1, result_type='broadcast')
percentages.plot(kind='barh', stacked=True, title='Findings distribution by gender')


plt.show()

#%%
# Finding incidence distribution by age
age_data = pd.melt(labels,
                   id_vars=['Patient Age'],
                   value_vars=findings_list + ['No Finding'],
                   var_name='Finding',
                   value_name='Count')
age_data = age_data.loc[age_data.Count>0]

#%%

for label in findings_list + ['No Finding']:
    plt.figure(figsize=(15, 10))
    ax1 = plt.subplot()
    sns.distplot(age_data[age_data['Finding'] == label]['Patient Age'], ax=ax1, bins=range(18, 95)[0::2])
    ax1.set(xlabel='Age', ylabel='Incidence')
    ax1.set_title(label, fontsize=18)
    #plt.show()
    plt.savefig('plots/' + label + '_age_distribution_plot.png', format='png')

#%%
# Correlation matrix
labels_for_corr = labels[['Patient Age', 'Patient Gender'] + findings_list]
labels_for_corr = pd.get_dummies(labels_for_corr, columns=['Patient Gender'])

x = labels_for_corr['Patient Age']
labels_for_corr['Patient Age'] = (x - x.min())/(x.max() - x.min())

corr = labels_for_corr.corr()