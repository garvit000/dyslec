#!/usr/bin/env python
# coding: utf-8

# # Deep learning Model of EEG Data Analysis 
# ### *This notebook was done as part of project in **"Introduction to BioInformatics** (BIO F242)"*

# ## Importing Libraries

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## *Reading* the datasets
# 
# ### What Dataset are we using ?
# https://www.kaggle.com/datasets/wanghaohan/confused-eeg
# 
# ### What does confusion have anything to do with Dyslexia?
# * Confusion is an important factor to detect Dyslexia
# * When dyslexic people make mistakes in reading or spelling, it is because they are experiencing disorientation
# * The person’s threshold for confusion is a key factor in how often he or she disorients.

# In[2]:


df=pd.read_csv('../input/confused-eeg/EEG_data.csv')
data = pd.read_csv('../input/confused-eeg/demographic_info.csv')
pd.read_csv('../input/confused-eeg/EEG_data.csv')


# ## Preprocessing the Data for running in Deep learning framework
# #### Merging the datasets on Subject-ID and editting (some of the columns-names had some extra space before them)

# In[3]:


data = data.rename(columns = {'subject ID': 'SubjectID',' gender':'gender',' age':'age',' ethnicity':'ethnicity'})
df = df.merge(data,how = 'inner',on = 'SubjectID')
df.head()


# ### Further processing to make the dataset proper before running through our model
# 1. See the shape
# 2. Find the missing columns
# 3. Convert the String data-type columns to Numerical ones
# 4. Check for imbalanced data-sets

# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.columns


# #### Converting the Categorical columns to numerical ones

# In[7]:


df['gender']=df['gender'].replace({'M':1,'F':0})
df['ethnicity']=df['ethnicity'].replace({'Han Chinese':0,'Bengali':1,'English':2})


# In[8]:


df.head()


# In[9]:


df['VideoID'].value_counts()


# #### Checking for *Imbalanced-dataset*

# In[10]:


df['predefinedlabel'].value_counts()


# In[11]:


for col in df.columns:
    if(df[col].isnull().sum()>0):
        print(col)


# In[12]:


df.describe()


# ## Identifying Target Variable : **Attention vs Mediation**
# 
# ### What is Mediation?
# * ##### Active Learning vs Passive Learning
# * ##### Mediated Learning Experience
# * ##### Absent Property in Confused Students

# ## Feature Identification
# ####  *EDA* to Identify the Feature-columns that are aligned with our *Target*-class

# In[13]:


sns.set_style('darkgrid')
sns.displot(data=df,x='Mediation',kde=True,aspect=16/7)


# #### Analyzing direct interference between Mediation and other features

# In[14]:


fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Attention',hue='user-definedlabeln')


# In[15]:


fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Raw',hue='user-definedlabeln')


# In[16]:


fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Theta',hue='user-definedlabeln')


# In[17]:


fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Alpha1',hue='user-definedlabeln')


# In[18]:


fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Gamma1',hue='user-definedlabeln')


# ## No *direct* interference has been identified between Features and Target-variable

# #### Importing library to perform Feature-Selection 

# In[19]:


from sklearn.feature_selection import mutual_info_classif


# ### Separating-out feature-set and `Target-column` 

# ### Mutual-info gives the score to each **Feature** which describes its *Relationship* with `Target` variable

# In[20]:


y=pd.get_dummies(df['user-definedlabeln'])
mi_score=mutual_info_classif(df.drop('user-definedlabeln',axis=1),df['user-definedlabeln'])
mi_score=pd.Series(mi_score,index=df.drop('user-definedlabeln',axis=1).columns)
mi_score=(mi_score*100).sort_values(ascending=False)
mi_score


# ### Selecting top-14 features

# In[21]:


mi_score.head(14).index


# In[22]:


top_fea=['VideoID', 'Attention', 'Alpha2', 'Delta', 'Gamma1', 'Theta', 'Beta1',
       'Alpha1', 'Mediation', 'Gamma2', 'SubjectID', 'Beta2', 'Raw', 'age']


# # Scaling our *Feature*-set

# In[23]:


from sklearn.preprocessing import StandardScaler
df_sc=StandardScaler().fit_transform(df[top_fea])


# # Importing libraries to build **Neural-Network**

# In[24]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks,layers


# ### *Splitting* the dataset into:
# 1. Training_Features, Training_Target
# 2. Testing_Features, Testing_Target
# 3. Validation_Features, Validation_Target

# In[25]:


from sklearn.model_selection import train_test_split
Xtr,xte,Ytr,yte=train_test_split(df_sc,y,random_state=108,test_size=0.27)
xtr,xval,ytr,yval=train_test_split(Xtr,Ytr,random_state=108,test_size=0.27)


# ## Running the Training Data into our Neural Network model

# In[26]:


# Model-Building step, stacking the hidden layers
model=keras.Sequential([
    layers.Dense(64,input_shape=(14,),activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.27),
    layers.Dense(124,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(248,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.32),   
    layers.Dense(512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.27),   
    layers.Dense(664,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.32),
    layers.Dense(264,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.27),
    layers.Dense(124,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(2,activation='sigmoid')
])
#Compiling the model with Adamax Optimizer
model.compile(optimizer='adamax',loss='binary_crossentropy',metrics='accuracy')


# ### Validating the model through Validation data

# In[27]:


#Creating the callback feature to stop the training in-Between, in case of no improvement
call=callbacks.EarlyStopping(patience=20,min_delta=0.0001,restore_best_weights=True)
#Fitting the model
history=model.fit(xtr,ytr,validation_data=(xval,yval),batch_size=28,epochs=150,callbacks=[call])


# ## Testing Accuracy of the Model

# In[28]:


model.evaluate(xte,yte)


# In[29]:


training=pd.DataFrame(history.history)


# ### Plotting the history of Neural-Network

# In[30]:


training.loc[:,['loss','val_loss']].plot()


# In[31]:


training.loc[:,['accuracy','val_accuracy']].plot()


# # How to Use this Model for Identifying Dyslexia ?
# 
# * #### When we get EEG data of people who are made to watch educational videos, we can get their RAW EEG signal and various frequency signals that it is composed of. 
# * #### We can then use this to predict the Mediation/Attention of the person. 
# * #### If we have enough dyslexic people in our training data sample, we can identify a cut off limit of the target feature and hence would be able to perform Diagnosis of Dyslexia based on whether the person's Mediation/Attention lies outside the cut-off limit.
