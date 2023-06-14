#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Da usare se ci sono problemi
#!pip install protobuf==3.20.3


# In[2]:


#Se ci sono problemi fai il restart del kernel
#pip install --upgrade pip


# In[3]:


#pip install scikit-learn
#pip install tensorflow


# In[5]:


import tensorflow as tf
print(tf.version.VERSION)


# In[6]:


#check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
#Show which devices the operations and tensors are assigned to. Enabling device placement logging causes any Tensor allocations or operations to be printed.
#tf.debugging.set_log_device_placement(True)


# In[7]:


#!pip install tensorflow-datasets


# In[8]:


#!pip install numpy


# In[9]:


#!pip install pandas


# In[10]:


#!pip install matplotlib


# In[11]:


import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd


# In[12]:


#LOADING AND PREPROCESSING


# In[13]:


#import the binary alpha digits dataset and split into train set and test set
import tensorflow_datasets as tfds


# In[ ]:


#Dà un avviso, nel caso poi vedere
(ds_nontest_orig, ds_test_orig), info = tfds.load('eurosat', split=['train[:78%]', 'train[78%:]'] ,with_info=True, shuffle_files=False)


# In[15]:


totallen=len(ds_nontest_orig)+len(ds_test_orig)
nontestlen=len(ds_nontest_orig)
testlen=len(ds_test_orig)
print("total dataset size: ", totallen)
print("non-test set size: ", nontestlen)
print("test set size: ", testlen)
print(info)


# In[16]:


trainlen=nontestlen*90//100  #consideriamo suddivisione non-test set 90:10 tra train set e validation set
validlen=nontestlen-trainlen
print("train set size: ", trainlen)
print("validation set size: ", validlen)
BUFFERSIZE=trainlen
while True:
    try:
        BATCHSIZE=int(input("Inserire batch size di training e validation set (intero, ad esempio 30): "))
        break
    except ValueError:
        print("Numero inserito non valido. Riprovare.")
    if BATCHSIZE<=0:
        raise ValueError("Numero inserito negativo o zero. Riprovare.\n")
ds_nontest=ds_nontest_orig.map(lambda item: (tf.cast(item['image'], tf.float32)/255.0, tf.cast(item['label'], tf.float32))).shuffle(buffer_size=BUFFERSIZE, reshuffle_each_iteration=False)
ds_test=ds_test_orig.map(lambda item: (tf.cast(item['image'], tf.float32)/255.0, tf.cast(item['label'], tf.float32))).batch(BATCHSIZE)
ds_train = ds_nontest.take(trainlen).batch(BATCHSIZE)
ds_valid = ds_nontest.skip(trainlen).batch(BATCHSIZE)


# In[17]:


#Visualizziamo N=25 immagini, con il rispettivo label
N=25
ds_for_pics=ds_nontest_orig.map(lambda item: (item['image'], item['label'])).take(N)
print("Salvataggio di %d immagini del dataset con rispettivo label." %N)
fig = plt.figure(figsize=(15, 12))                #modifica se N viene modificato
for i,(image,label) in enumerate(ds_for_pics):
    ax = fig.add_subplot(5, 5, i+1)               #modifica se N viene modificato
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    if label==0:
        string='AnnualCrop'
    elif label==1:
        string='Forest'
    elif label==2:
        string='HerbaceousVegetation'
    elif label==3:
        string='Highway'
    elif label==4:
        string='Industrial'
    elif label==5:
        string='Pasture'
    elif label==6:
        string='PermanentCrop'
    elif label==7:
        string='Residential'
    elif label==8:
        string='River'
    else:
        string='SeaLake'
    ax.set_title('{:s} ({})'.format(string, label), size=10)
plt.savefig("sample_from_dataset.pdf", format="pdf", bbox_inches="tight")


# In[18]:


#IMPLEMENTING CNN


# In[19]:


while True:
    try:
        NUMEPOCHS=int(input("Inserire numero di epochs (intero, ad esempio 20): "))
        break
    except ValueError:
        print("Numero inserito non valido. Riprovare.")
    if NUMEPOCHS<=0:
        raise ValueError("Numero inserito negativo o zero. Riprovare.\n")


# In[20]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='swish'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(rate=0.5),
   
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='swish'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(rate=0.5),
    
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='swish'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(rate=0.5),
   
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='swish'),
    tf.keras.layers.MaxPooling2D((2, 2)),
   
    tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='swish')
])


# In[21]:


model.compute_output_shape(input_shape=(None, 64, 64, 3))


# In[22]:


model.add(tf.keras.layers.GlobalAveragePooling2D())
model.compute_output_shape(input_shape=(None, 64, 64, 3))


# In[23]:


model.add(tf.keras.layers.Dense(10, activation='softmax'))
tf.random.set_seed(1)
model.build(input_shape=(None, 64, 64, 3))
model.summary()


# In[24]:


model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])


# In[25]:


hist = model.fit(ds_train, validation_data=ds_valid, epochs=NUMEPOCHS, shuffle=True)


# In[26]:


model.save('eurosat_classifier.h5', overwrite=True, include_optimizer=True, save_format='h5')

# In[ ]:


print(hist.history.keys())

# In[27]:


history = hist.history
print("Produzione e salvataggio dei plot di loss e accuracy")
x_arr = np.arange(len(history['loss'])) + 1
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, history['loss'], '-o', label='Train loss')
ax.plot(x_arr, history['val_loss'], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, history['accuracy'], '-o', label='Train acc.')
ax.plot(x_arr, history['val_accuracy'], '--<',label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)
plt.savefig("plots_loss_acc.pdf", format="pdf", bbox_inches="tight")


# In[ ]:


#nel caso in cui vogliamo continuare con altre epochs
while True:
    try:
        choice=int(input("Inserire 1 se si desidera continuare il training con epochs aggiuntive, 0 se non si desidera aggiungere ulteriori epochs: "))
        break
    except ValueError:
        print("Scelta inserita non valida. Riprovare.")
    if (choice!=0)and(choice!=1):
        raise ValueError("Scelta inserita non valida. Riprovare.\n")
if choice==1:
    while True:
        try:
            ADDITIONALEPOCHS=int(input("Inserire numero di epochs aggiuntive: "))
            break
        except ValueError:
            print("Numero inserito non valido. Riprovare.")
        if ADDITIONALEPOCHS<=0:
            raise ValueError("Numero inserito negativo o zero. Riprovare.\n")
    hist = model.fit(ds_train, validation_data=ds_valid, epochs=ADDITIONALEPOCHS, initial_epoch=NUMEPOCHS, shuffle=True)
    
    model.save('eurosat_classifier_augmented.h5', overwrite=True, include_optimizer=True, save_format='h5')
    
    history = hist.history
    print("Produzione e salvataggio dei plot di loss e accuracy")
    x_arr = np.arange(len(history['loss'])) + 1
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, history['loss'], '-o', label='Train loss')
    ax.plot(x_arr, history['val_loss'], '--<', label='Validation loss')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, history['accuracy'], '-o', label='Train acc.')
    ax.plot(x_arr, history['val_accuracy'], '--<',label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)
    plt.savefig("plots_loss_acc_augmented.pdf", format="pdf", bbox_inches="tight")
    


# In[ ]:


scores = model.evaluate(ds_test, verbose=0)


# In[ ]:


print("Test loss : {}".format(scores[0]))
print("Test accuracy: {}".format(scores[1]))
outfile=open("test_loss_acc.txt", "w")
outfile.write("Test loss : {}\n".format(scores[0]))
outfile.write("Test accuracy: {}\n".format(scores[1]))
outfile.close()


# In[ ]:


#Vediamo i risultati per M=25 elementi del test set
M=25

predicted_probabilities=model.predict(ds_test)  #returns a numpy array
tf.print(predicted_probabilities.shape)
predicted_label=tf.math.argmax(predicted_probabilities, axis=1).numpy()  #axis is the axis to reduce across. Default to 0.
label_probabilities=np.zeros(len(predicted_label))
print("Predicted probabilities vector length: %d" %len(predicted_label))
for k in range(len(predicted_label)):
    label_probabilities[k]=predicted_probabilities[k][predicted_label[k]] 

print("Visualizziamo i risultati relativi a %d immagini del test set." %M)
ds_results=ds_test.unbatch()
ds_M_results = ds_results.take(M)

fig_t_1 = plt.figure(figsize=(15, 12))             #modifica se M viene modificato
for i,(image,label) in enumerate(ds_M_results):
    ax = fig_t_1.add_subplot(5, 5, i+1)            #modifica se M viene modificato
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(image)
    if label==0:
        true_string='AnnualCrop'
    elif label==1:
        true_string='Forest'
    elif label==2:
        true_string='HerbaceousVegetation'
    elif label==3:
        true_string='Highway'
    elif label==4:
        true_string='Industrial'
    elif label==5:
        true_string='Pasture'
    elif label==6:
        true_string='PermanentCrop'
    elif label==7:
        true_string='Residential'
    elif label==8:
        true_string='River'
    else:
        true_string='SeaLake'
    ax.set_title('{:s} ({:d})'.format(true_string, int(label)), size=10)
    if predicted_label[i]==0:
        predicted_string='AnnualCrop'
    elif predicted_label[i]==1:
        predicted_string='Forest'
    elif predicted_label[i]==2:
        predicted_string='HerbaceousVegetation'
    elif predicted_label[i]==3:
        predicted_string='Highway'
    elif predicted_label[i]==4:
        predicted_string='Industrial'
    elif predicted_label[i]==5:
        predicted_string='Pasture'
    elif predicted_label[i]==6:
        predicted_string='PermanentCrop'
    elif predicted_label[i]==7:
        predicted_string='Residential'
    elif predicted_label[i]==8:
        predicted_string='River'
    else:
        predicted_string='SeaLake'
                            
    if predicted_label[i]==label:
       ax.text(0.5, -0.15, '{:s} ({:d})\nProb={:.0f}%'.format(predicted_string, predicted_label[i], label_probabilities[i]*100), size=16, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=9, fontweight='bold', color='green')
    else:
       ax.text(0.5, -0.15, '{:s} ({:d})\nProb={:.0f}%'.format(predicted_string, predicted_label[i], label_probabilities[i]*100), size=16, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=9, fontweight='bold', color='red')
plt.tight_layout()
plt.savefig("samples_from_test.pdf", format="pdf", bbox_inches="tight")


# In[ ]:


#Plotting the first K=25 mismatched images along with their predictions
K=25
print("Visualizziamo i primi %d elementi non classificati correttamente del test set." %K)
fig_t_2=plt.figure(figsize=(15, 12))                  #modifica se K viene modificato
counter=0
for i,(image,label) in enumerate(ds_results):
    if predicted_label[i]!=label:  
        ax = fig_t_2.add_subplot(5, 5, counter+1)       #modifica se K viene modificato
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(image)
        if label==0:
            true_string='AnnualCrop'
        elif label==1:
            true_string='Forest'
        elif label==2:
            true_string='HerbaceousVegetation'
        elif label==3:
            true_string='Highway'
        elif label==4:
            true_string='Industrial'
        elif label==5:
            true_string='Pasture'
        elif label==6:
            true_string='PermanentCrop'
        elif label==7:
            true_string='Residential'
        elif label==8:
            true_string='River'
        else:
            true_string='SeaLake'
        ax.set_title('{:s} ({:d})'.format(true_string, int(label)), size=10)
        if predicted_label[i]==0:
            predicted_string='AnnualCrop'
        elif predicted_label[i]==1:
            predicted_string='Forest'
        elif predicted_label[i]==2:
            predicted_string='HerbaceousVegetation'
        elif predicted_label[i]==3:
            predicted_string='Highway'
        elif predicted_label[i]==4:
            predicted_string='Industrial'
        elif predicted_label[i]==5:
            predicted_string='Pasture'
        elif predicted_label[i]==6:
            predicted_string='PermanentCrop'
        elif predicted_label[i]==7:
            predicted_string='Residential'
        elif predicted_label[i]==8:
            predicted_string='River'
        else:
            predicted_string='SeaLake'
        ax.text(0.5, -0.15, '{:s} ({:d})\nProb={:.0f}%'.format(predicted_string, predicted_label[i], label_probabilities[i]*100), size=16, horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=9, fontweight='bold', color='red')
        counter += 1
    if counter == K: 
        break
plt.tight_layout()
plt.savefig("samples_mismatched_test.pdf", format="pdf", bbox_inches="tight")

