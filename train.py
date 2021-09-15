import numpy as np
from encoder_decoder import Resnet3DBuilder1
import pickle
import os
import math
import tensorflow as tf
from numpy import float32
from keras.optimizers import Adam,SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from load_data import *
from keras.models import load_model,Model
from scipy.signal import find_peaks
import keras
from keras.losses import categorical_crossentropy
from gendata_new import *
from keras.layers import Lambda,Input,Embedding
import keras.backend as K

config = ConfigProto()
config.gpu_options.allow_growth = True   #动态申请显存
session = InteractiveSession(config=config)

nb_epoch = 100  # number of epoch at training stage
step=300
batch_size = 2 # batch size
val_batch=2
val_step=50
lr = 1e-4
num_class=18

#filename1=generate_filename(image_path="E:/PURE_DATA/real_face_emotion/")
#filename2=generate_filename(image_path="E:/PURE_DATA/artificial_subject/subject2/")
filename3=generate_filename(image_path="E:/PURE_DATA/64_data/predict_check1/")#real_face_emotion
filename4=generate_filename(image_path='E:/PURE_DATA/64_data/subject1/')#64_data/predict_90,64_data/subject1
#random.shuffle(filename6)
#filename_train=filename1[22000:26000]#+filename2[0:2500]
filenameval=filename4#+filename4

model = Resnet3DBuilder1.encoder_decoder((64, 64, 64, 3))
sgd=SGD(lr=lr, momentum=0.9, nesterov=True)
adm=Adam(lr=lr,decay=0.000002)
def l_im(y_true,y_pred,e=1):
    #print('y_pred,y_true:',K.mean(y_true,axis=2).shape)
    #p=K.mean(K.square(K.sum(y_true,axis=(2,3,4))-K.sum(y_pred,axis=(2,3,4))))
    #p=1-keras.losses.cosine_similarity(K.mean(y_true,axis=(2,3,4)),K.mean(y_pred,axis=(2,3,4)))
    #p =1- keras.losses.cosine_similarity((K.mean(y_true, axis=(2, 3, 4))-K.mean(y_true)), (K.mean(y_pred, axis=(2, 3, 4))-K.mean(y_pred)))
    mae=K.mean(K.abs(y_pred-y_true))
    mse=K.mean(K.square(y_pred-y_true))
    psnr=(10*K.log(65025/(mse)))/2.303
    s=tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

    return 100*mae+s#+s#50*mae+20*mse+s

def l_p(y_true,y_pred):
    rmse=K.sqrt(K.mean(K.square(y_pred-y_true)))
    #p = 1 - keras.losses.cosine_similarity((y_pred - K.mean(y_true,axis=(1,2,3,4))),(y_pred - K.mean(y_true,axis=(1,2,3,4))))
    p=1 + keras.losses.cosine_similarity(y_pred, y_true)
    #y_t =K.mean(y_true, axis=1)
    #y_p = K.mean(y_pred, axis=1)
    #pear = keras.losses.cosine_similarity(K.minimum(y_pred , y_p), K.minimum(y_true , y_t))
    return rmse

def m_im(y_true,y_pred):
    return 50*K.mean(K.square(y_pred-y_true))

def m_p(y_true,y_pred):
    p1 = 1+keras.losses.cosine_similarity(y_pred,y_true)
    return p1

losses={'activation_25':l_im}#
metrics={'activation_25':m_im}#'activation_24':m_im,
weight={'activation_25':1}
model.compile(optimizer=adm, loss=losses,loss_weights=weight,metrics=metrics)#loss_weights=weight,
model.summary()

#CENTER-LOSS
#if isCenterloss:
lambda_c = 0.3
feature_size=18
from keras.callbacks import EarlyStopping, ModelCheckpoint
hyperparams_name = 'lr{}'.format(lr)
fname_param = 'supress_noise_1.75%-ubfc.h5'.format(hyperparams_name)

early_stopping = EarlyStopping(monitor='val_loss', patience=10,mode='min')
model_checkpoint = ModelCheckpoint(fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
model.load_weights('D:\python_code\keras-resnet3d-master\oise-supress\supress_noise_1.75%-ubfc.h5',by_name=True)#lr0.001.BNmain_rmse_complex.h5

print('=' * 10)
print("training model...")

history=model.fit_generator(generator = load_data(batch_size,step,18),
          steps_per_epoch = step,
          epochs = nb_epoch,
          verbose = 1,
          callbacks =[early_stopping, model_checkpoint],
          shuffle=  True,
          validation_data=load_data(batch_size, val_step, 18),
          validation_steps=val_step)

'''history=model.fit_generator(generator = read_data(filename_train,batch_size),
          steps_per_epoch = (len(filename_train) + batch_size - 1) // batch_size,
          epochs = nb_epoch,
          verbose = 1,
          callbacks =[early_stopping, model_checkpoint],
          shuffle=  True,
          validation_data = read_data(filenameval,val_batch),
          validation_steps=len(filenameval)//val_batch)
          validation_data = load_data(batch_size,val_step,18),
          validation_steps=val_step
'''

model.save_weights('{}.BNmain_rmse.h5'.format(hyperparams_name), overwrite=True)
model.save('0.001_model.h5')
history_dict=history.history
train_loss= history_dict['loss']
val_loss = history_dict['val_loss']
train_acc= history_dict['activation_25_m_im']
val_acc = history_dict['val_activation_25_m_im']
import matplotlib.pyplot as plt
#绘制损失曲线
plt.figure()
plt.plot(range(len(train_loss)), train_loss, label="train_loss")
plt.plot(range(len(val_loss)), val_loss, label="val_loss")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('train')
plt.savefig('./loss.jpg')
plt.show()
plt.figure()
plt.plot(range(len(train_loss)), train_acc, label="train_acc")
plt.plot(range(len(val_loss)), val_acc, label="val_acc")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('train')
plt.savefig('./accuracy.jpg')
plt.show()
print('train_loss:',train_loss)
print('val_loss:',val_loss)
print('train_acc:',train_acc)
print('val_acc:',val_acc)

print('train_loss:',list(np.float16(train_loss)))
print('val_loss:',list(np.float16(val_loss)))
print('train_acc:',list(np.float16(train_acc)))
print('val_acc:',list(np.float16(val_acc)))

