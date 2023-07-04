#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
# In[2]:
import keras
# In[3]:
import cv2
import os
from PIL import Image
# In[4]:
import tensorflow as ts
# In[5]:
from matplotlib import pyplot as plt
# In[6]:
import pandas as pd
# In[7]:
np.random.seed(1000)
# In[8]:
import numpy as np
# In[9]:
os.environ['KERAS_BACKEND']='tensorflow'
# # Model Building
# In[10]:
image_directory=(r'C:\Users\cm20470\Anu_python/')
# In[11]:
SIZE=64
# In[12]:
dataset=[]
label=[]
# In[13]:
Pseparated_images=os.listdir(image_directory + 'Pseparated/')
# In[14]:
Pseparated_images
# In[15]:
for i, image_name in enumerate(Pseparated_images):
    if (image_name.split('.')[1]=='png'):
        image=cv2.imread(image_directory + 'Pseparated/' + image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        label.append(1)

# In[16]:
len(dataset)
# In[17]:
Dispersed_images=os.listdir(image_directory + 'Dispersed/')
# In[18]:
for i, image_name in enumerate(Dispersed_images):
    if (image_name.split('.')[1]=='png'):
        image=cv2.imread(image_directory + 'Dispersed/' + image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((SIZE,SIZE))
        dataset.append(np.array(image))
        label.append(0)

# In[19]:
INPUT_SHAPE=(SIZE,SIZE,3)
# In[20]:
inp=keras.layers.Input(shape=INPUT_SHAPE)
# In[21]:
conv1=keras.layers.Conv2D(32,kernel_size=(3,3),
                          activation='relu',
                          padding='same')(inp)

# In[22]:
pool1=keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
norm1=keras.layers.BatchNormalization(axis=-1)(pool1)
drop1=keras.layers.Dropout(rate=0.2)(norm1)
# In[23]:
conv2=keras.layers.Conv2D(32,kernel_size=(3,3),
                          activation='relu',
                          padding='same')(drop1)

# In[24]:
pool2=keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
norm2=keras.layers.BatchNormalization(axis=-1)(pool2)
drop2=keras.layers.Dropout(rate=0.2)(norm2)
# In[25]:
flat=keras.layers.Flatten()(drop2)
# In[26]:
hidden1=keras.layers.Dense(512,activation='relu')(flat)
norm3=keras.layers.BatchNormalization(axis=-1)(hidden1)
drop3=keras.layers.Dropout(rate=0.2)(norm3)
# In[27]:
hidden2=keras.layers.Dense(512,activation='relu')(drop3)
norm4=keras.layers.BatchNormalization(axis=-1)(hidden2)
drop4=keras.layers.Dropout(rate=0.2)(norm4)
# In[28]:
out=keras.layers.Dense(2,activation='sigmoid')(drop4)
# In[29]:
model=keras.Model(inputs=inp,outputs=out)
# In[30]:
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# In[31]:
print(model.summary())
# In[44]:
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
# In[68]:
X_train, X_test, y_train, y_test = train_test_split(dataset, to_categorical(np.array(label)), test_size=0.20, random_state=1)
# In[69]:
history=model.fit(np.array(X_train),y_train, batch_size=5,verbose=1,epochs=8,validation_split=0.2,shuffle=True)
# In[65]:
print("Test_Accuracy: {:.1f}%".format(model.evaluate(np.array(X_test),np.array(y_test))[1]*100))
# In[63]:
f,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
t=f.suptitle('CNN Performance',fontsize=12)
f.subplots_adjust(top=0.85,wspace=0.3)
max_epoch=len(history.history['accuracy']) + 1
epoch_list=list(range(1,max_epoch))
ax1.plot(epoch_list,history.history['accuracy'],label='Train Accuracy')
ax1.plot(epoch_list,history.history['val_accuracy'],label='Validation Accuracy')
ax1.set_xticks(np.arange(1,max_epoch,5))
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy value')
ax1.set_title('Accuracy')
l1=ax1.legend(loc='best')
ax2.plot(epoch_list,history.history['loss'],label='Train Loss')
ax2.plot(epoch_list,history.history['val_loss'],label='Validation Loss')
ax2.set_xticks(np.arange(1,max_epoch,5))
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss value')
ax2.set_title('Loss')
l2=ax2.legend(loc='best')
# In[119]:
model.save('phase_classification.h5')
# In[ ]:




