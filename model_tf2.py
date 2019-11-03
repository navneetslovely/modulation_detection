import os,random
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')
import seaborn as sns
import pickle as pic
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.models as models
import tensorflow.keras.layers

from tensorflow.keras.layers import Input, LSTM, RepeatVector,TimeDistributed, MaxPool2D, ZeroPadding2D, Conv2D, PReLU, Reshape,Dense,Dropout,Activation, Flatten, BatchNormalization
from tensorflow.keras.regularizers import *
# In[2]:

'''getting the database'''

with open("RML2016.10a_dict.pkl",'rb') as db:  #
    ns=pic.load(db,encoding='latin1')
sig_nos_ratio,modu=map(lambda j:sorted(list(set(map(lambda x:x[j],ns.keys())))),[1,0])

data_db=[]
data_lbl=[]
for mod in modu:
    for snr in sig_nos_ratio:
        data_db.append(ns[(mod,snr)])
        for i in range(ns[(mod,snr)].shape[0]): data_lbl.append((mod,snr))
data_db=np.vstack(data_db)  # vertical axes stacking the data

"""partition the sample dataset into blocks : training block and test block """

np.random.seed(2016)
n_sample=data_db.shape[0]
n_train=n_sample * 0.5
train_index=np.random.choice(range(0,n_sample),size=int(n_train),replace=False)
test_index=list(set(range(0,n_sample)) - set(train_index))
data_db_train=data_db[train_index]
data_db_test=data_db[test_index]


def convert_onehot(binary_vector):
    binary_vector=list(binary_vector)
    binary_vectors=np.zeros([len(binary_vector),max(binary_vector) + 1])
    binary_vectors[np.arange(len(binary_vector)),binary_vector]=1
    return binary_vectors


new_data_db_train=convert_onehot(map(lambda x:modu.index(data_lbl[x][0]),train_index))
new_data_db_test=convert_onehot(map(lambda x:modu.index(data_lbl[x][0]),test_index))

new_train_index=[np.where(row == 1)[0][0] for row in new_data_db_train]
new_test_index=[np.where(row == 1)[0][0] for row in new_data_db_test]

# n_sample

data_shape=list(data_db_train.shape[1:])
print(data_db_train.shape,data_shape)
type_modu=modu

"""model initialization"""

model_exists=False
if model_exists:
    del model
# dropout rate for the weights in the models in percentage
dropout_rate=0.4

model=models.Sequential()
model.add(Reshape([1] + data_shape,input_shape=data_shape))

# layer_1
model.add(ZeroPadding2D((0,2),data_format='channels_first'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(1,16),padding="valid",name="con_layer_1",data_format='channels_first'))
model.add(PReLU())
model.add(MaxPool2D(pool_size=(1,2),data_format='channels_first'))

# layer_2
model.add(ZeroPadding2D((0,2),data_format='channels_first'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(1,16),padding="valid",name="con_layer_2",data_format='channels_first'))
model.add(PReLU())
# model.add(MaxPool2D(pool_size=(1,2),data_format='channels_first'))

# layer_3
model.add(ZeroPadding2D((0,2),data_format='channels_first'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(1,16),padding="valid",name="con_layer_3",data_format='channels_first'))
model.add(PReLU())
# model.add(MaxPool2D(pool_size=(1,2),data_format='channels_first'))

# layer_4
# model.add(ZeroPadding2D((0, 2), data_format= 'channels_first'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(1,16),padding="valid",name="con_layer_4",data_format='channels_first'))
model.add(PReLU())
# model.add(MaxPool2D(pool_size=(1,2),data_format='channels_first'))

model.add(Dropout(dropout_rate))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Reshape((1,model.output_shape[1])))

model.add(LSTM(32,return_sequences=False))

model.add(Dense(len(type_modu),kernel_initializer="he_normal",name="dense1"))
model.add(Activation('softmax'))

model.add(Reshape([len(type_modu)]))
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.summary()
model_exists=True

# number of epochs to train
number_epoch=150
# batch size of training
batch_size=1024

filepath='convmodrecnets_CNN2_0.5.wts.h5'

history = model.fit(data_db_train, new_data_db_train, batch_size= batch_size,epochs= number_epoch,verbose=2,
                          validation_data= (data_db_train, new_data_db_train),
                          callbacks = [ tf.keras.callbacks.ModelCheckpoint(filepath, monitor= 'val_loss',verbose = 0 , only_save_best= True, mode= 'auto'),
                                        tf.keras.callbacks.EarlyStopping(monitor='val_loss' , patience= 5, verbose =0 , mode='auto'),
                          ])
model.load_weights(filepath)

performance = model.evaluate(data_db_test,new_data_db_test, verbose=0, batch_size= batch_size)
print(performance)


plt.figure()
plt.title('Performance of Training')
plt.plot(history.epoch,history.model.history['loss'],label='train loss+error')
plt.plot(history.epoch, history,history['val_loss'], label='error_value')
plt.legend()

def confusion_matrix(con_matrix, title= 'Confusion Matrix',conmatr=plt.con_matrix.Blues, labels=[]):
    plt.imshow(con_matrix,interpolation='nearest',conmatr=conmatr)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(labels))
    plt.xticks(tick_marks,labels,rotation=45)
    plt.yticks(tick_marks,labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Plot confusion matrix
test_Y_hat = model.predict(data_db_test, batch_size=batch_size)
conf = np.zeros([len(type_modu),len(type_modu)])
confnorm = np.zeros([len(type_modu),len(type_modu)])
for i in range(0,data_db_test.shape[0]):
    j = list(new_data_db_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(type_modu)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
confusion_matrix(confnorm, labels=type_modu)

acc={}
for snr in sig_nos_ratio:

    # extract classes @ SNR
    test_sig_nos_ratio=list(map(lambda x:data_lbl[x][1],train_index))
    test_X_i=data_db_test[np.where(np.array(test_sig_nos_ratio) == snr)]
    test_Y_i=new_data_db_test[np.where(np.array(test_sig_nos_ratio) == snr)]

    # estimate classes
    test_Y_i_hat=model.predict(test_X_i)
    conf=np.zeros([len(type_modu),len(type_modu)])
    confnorm=np.zeros([len(type_modu),len(type_modu)])
    for i in range(0,test_X_i.shape[0]):
        j=list(test_Y_i[i,:]).index(1)
        k=int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k]=conf[j,k] + 1
    for i in range(0,len(type_modu)):
        confnorm[i,:]=conf[i,:] / np.sum(conf[i,:])
    #     plt.figure()
    #     plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))

    cor=np.sum(np.diag(conf))
    ncor=np.sum(conf) - cor
    print("Overall Accuracy: ",cor / (cor + ncor))
    acc[snr]=1.0 * cor / (cor + ncor)
# %%
# Plot accuracy curve
plt.plot(sig_nos_ratio,list(map(lambda x:acc[x],sig_nos_ratio)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")

# In[17]:


# Save results to a pickle file for plotting later
print(acc)
fd=open('results_cnn2_d0.5.dat','wb')
pic.dump(("CNN2",0.5,acc),fd)