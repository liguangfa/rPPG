import cv2
import numpy as np
from keras.optimizers import Adam,SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from load_data import *
from gendata_new import *
from keras.models import load_model ,Model
import matplotlib.pyplot as plt
from encoder_decoder import Resnet3DBuilder1
import random

nb_epoch = 100  # number of epoch at training stage
nb_epoch_cont = 5  # number of epoch at training (cont) stage
batch_size = 4 # batch size
batch_size_evlu=16
lr = 0.0001  # learning rate

config = ConfigProto()
config.gpu_options.allow_growth = True  # 动态申请显存
session = InteractiveSession(config=config)

sgd = SGD(lr=lr)
adm = Adam(lr=lr, decay=0.000002)
model = Resnet3DBuilder1.encoder_decoder((64, 64, 64, 3))
model.summary()  # Prints a string summary of the network
model.load_weights('D:\python_code\keras-resnet3d-master\oise-supress\supress_noise_1.75%-2.h5',by_name=True)#lr0.0001.BNmain_rmse.h5,lr0.001.BNmain_rmse_subject1-1.5%.h5,lr0.001.BNmain_rmse_complex_noise-94%.h5,lr0.001.BNmain_rmse_complex_subject1.h5

filename1=generate_filename(image_path="E:/PURE_DATA/64_data/pre_sub4_64/")#predict_check1_64
filename2=generate_filename(image_path="E:/PURE_DATA/64_data/pre_sub8_64/")#pre_sub8_64,predict_rppg_64
filename3=generate_filename(image_path="E:/UBFC_DATASET/data_for_test/")
#random.shuffle(filename1)
#random.shuffle(filename3)
filename=filename1+filename2#+filename3#[20:]#+filename4[0:18]#[125:]
filenamex=filename1
l=len(filename)

pre_heart=[]
rppg_heart=[]
test_heart=[]
ssi=[]
print('1:',l)
a=1
p=[]
t=[]
fft_heart=[]


def calc_ssim(img1, img2):
  C1 = (0.01 * 255) ** 2
  C2 = (0.03 * 255) ** 2

  img1 = img1.astype(np.float64)
  img2 = img2.astype(np.float64)
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())

  mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
  mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
  mu1_sq = mu1 ** 2
  mu2_sq = mu2 ** 2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
  sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
  sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

  ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                          (sigma1_sq + sigma2_sq + C2))
  return ssim_map.mean()

for j in range(l):#
  filename_read = filename[a*j:a*j+a]
  X_test = read_ppg(filename_read)[0]
  Y_test = read_ppg(filename_read)[1:3]
  X_test=X_test/255
  #print('x_test:', Y_test[1])

  '''filename_input = filenamex[a * j:a * j + a]
  X_input = read_datavar(filename_input)[0]
  Y_input = read_datavar(filename_read)[1:3]
  X_input = X_input / 255
  X_input=X_input[0][0:64]
  X_fake=X_input[None,:,:,:,:]'''
  #print('x_test:',Y_test[0])
  test_heart.append(Y_test[0][0])

  '''x_gen=load_data(1,1,18)
  print(x_gen)'''

  input_test = {"input_1": X_test}
  layer_model = Model(inputs=model.input, outputs=model.output)  # outputs=model.layer[351].output
  feature = layer_model.predict(input_test)[1] #1-image
  feature0 = layer_model.predict(input_test)[0] #0-OUT
  feature1=np.squeeze(feature0)
  print('feature:',feature.shape,Y_test[0].shape,Y_test[0].shape)

  j = j + 1
  #save amplify
  path = 'E:/PURE_DATA/ALPHA=40/'
  #results = np.array(results, dtype="uint8")
  os.mkdir(path + str(int(j+98)))  # 创建一个按数字顺序的文件夹,/media/li/TOSHIBA EXT/rppgnet_train/rppd_train
  path1 = path + '/' + str(int(j+98)) + '/'
  file_image = h5py.File(path1 + '/map.h5', 'w')
  #results = np.array(results, dtype='uint8')
  file_image.create_dataset('map.h5', data=np.squeeze(feature))
  file_label = h5py.File(path1 + '/bpm.h5', 'w')
  file_label.create_dataset('bpm.h5', data=np.squeeze(Y_test[0])) # np.mean(x[((i//180)-1)*300:(i//180)*300])))
  '''file_label = h5py.File(path1 + '/rppg.h5', 'w')
  file_label.create_dataset('rppg.h5', data=np.squeeze(Y_test[1]))'''



  z=[]
  w=[]
  v=[]
  s = []
  p_pre=[]
  p_test=[]
  y = []
  x = []

  '''for i in range(64):

    image=feature[0,:,:,:][i]*255
    image_rppg = feature0[0, :, :, :][i] * 255
    image_original=X_test[0,:,:,:][i]*255
    #cv2.imshow('video',image.astype('uint8'))
    #cv2.waitKey(5)
    #cv2.imwrite('C:/Users/liguangfa/Desktop/experiment/image_time/'+'motion_rppg'+str(i)+'.png',image_rppg)
    x.append(np.mean(image[:,:,1][image[:,:,1]>20]))#[:,:,1][image[:,:,1]>10]
    z.append(np.mean(image[:, :, 2][image[:, :, 2] > 20]))
    y.append(np.mean(image_original[:,:,1][image_original[:,:,1]>20]))#[:,:,1][image_original[:,:,1]>10]
    w.append(np.mean(feature1[i]>0))
    #v.append(np.squeeze(Y_test[1])[::4])
    #label=np.squeeze(Y_test[0])[i]
    #p_pre.append(np.mean(label))
    p_test.append(np.mean(image_rppg))'''


  '''#ssim
    im1 = label
    im2 = image_rppg
    ssim = calc_ssim(im1, im2)
    s.append(ssim)'''
    #pearson

  #p.append(np.corrcoef(p_pre,p_test))
  #t.append(p_test)
  #print('p:',np.mean(p))#np.corrcoef(p,t))
  #print('t:', t)
  '''s1 = np.mean(np.array(s))
  ssi.append(s1)
  print('ssim:', np.mean(s), np.mean(ssi))'''



  '''r =np.squeeze(Y_test[0])[::2]
  #print('x:',x)
  xpeak_id, xpeak_property = find_peaks(r, distance=5, prominence=0.11)  # ,prominence=0.1
  #print('xfreq:', xpeak_id, xpeak_property)
  xinter = []
  if len(xpeak_id) == 1:
    xfreq = 40
  else:
    for h in range(1, min([len(xpeak_id)])):
      xper_inter = xpeak_id[h] - xpeak_id[h - 1]
      xinter.append(xper_inter)
    xfreq = 900 / np.mean(xinter)
    print('xfreq:', xfreq)
  heart_rate = xfreq
  rppg_heart.append(heart_rate)


  x1 = x#np.array(w)*100
  #print('x:',x1)
  xpeak_id1, xpeak_property1 = find_peaks(x1, distance=5, prominence=0.1)  # ,prominence=0.1
  # print('xfreq:', xpeak_id, xpeak_property)
  xinter1 = []
  if len(xpeak_id1) == 1:
    xfreq = 40
  else:
    for h in range(1, min([len(xpeak_id1)])):
      xper_inter1 = xpeak_id1[h] - xpeak_id1[h - 1]
      xinter1.append(xper_inter1)
    xfreq1 = 900 / np.mean(xinter1)
    print('xfreq1:', xfreq1)
  heart_rate1 = xfreq1
  pre_heart.append(heart_rate1)'''

  '''x = x
  fft = np.fft.fft(x)
  fftshift = np.fft.fftshift(fft)
  amp = abs(fftshift) / len(fft)
  amp = amp[range(len(amp) // 2)]
  pha = np.angle(fftshift)
  fre = np.fft.fftshift(np.fft.fftfreq(d=1 /15, n=len(x)))
  fre = fre[range(len(fre) // 2)]
  heart_fft = abs(fre[amp[1:].tolist().index(max(amp[1:]))])*60
  fft_heart.append(heart_fft)
  print('fft_heart:',heart_fft)'''

  #print('ori:',y)
  #print('amp:', x)
  #print('rppg:', list(np.float16(np.squeeze(Y_test[1])[::4])))

  '''plt.subplot(121)
  plt.plot(range(len(np.squeeze(x))), x, label="pred_G")  # [0:80],np.max(x)
  plt.plot(range(len(np.squeeze(y))), y, label="ori")  # [0:80],/np.max(y)
  #plt.plot(range(len(np.squeeze(z))), y / np.max(z), label="pred_b")
  #plt.plot(range(len(np.squeeze(Y_test[1])[::4])), np.squeeze(Y_test[1])[::4]+20, label="ppg")#/np.max(np.squeeze(Y_test[1])[::4])
  plt.legend()
  plt.subplot(122)
  plt.plot(range(len(w)), np.array(w)/0.5, label="pred")  # [0:80]
  #plt.plot(range(len(np.squeeze(Y_test[1])[::4])), np.squeeze(Y_test[1])[::4]/50, label="ppg")
  plt.legend()
  plt.xlabel('time')
  plt.ylabel('amplitutide')
  plt.title('wave')
  plt.savefig('./waveform1.jpg')
  plt.show()'''

print('test_heart:',test_heart)
print('fft_heart:',fft_heart)
print('rppg_heart:',rppg_heart)
print('pre_heart:',pre_heart)








