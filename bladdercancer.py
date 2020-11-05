import pandas as pd 
import numpy as np 
import dicom
import matplotlib.pylab as plt
import matplotlib.patches as mpatches import scipy as sp
import matplotlib as mpl import os
import SimpleITK as sitk

def remove_pixels(imgWhiteMatter9, line): 
    img_final = imgWhiteMatter9
    y_size = img_final.GetSize()[1]
    x_size = img_final.GetSize()[0]
    lst = []
    for j in range(0,y_size):
        lst_line = []
        for i in range(0,x_size): 
            lst_line.append(img_final.GetPixel(i,j))
        lst.append(lst_line)
        unique, counts = np.unique(lst[line], return_counts=True)
        dic = {}
        for z in range(0,len(unique)):
            dic[unique[z]] = counts[z]
        lst_order = []
        for w in sorted(dic, key=dic.get, reverse=True): 
            lst_order.append(w)
        
        for i in range(0,y_size): 
            for j in range(0,x_size):
                if img_final.GetPixel(j,i) != lst_order[1]:
                    img_final.SetPixel(j,i,0) 
        return img_final
    
def reject_outliers(data): 
    m = 2
    u = np.mean(data) 
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)] 
    return filtered
    
def sitk_show(img, title=None, margin=0.05, dpi=40, cmap="gray"): 
    nda = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1]*spacing[1], nda.shape[0]*spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])
    plt.set_cmap(cmap) 
    ax.imshow(nda,extent=extent,interpolation=None)
    if title:
        plt.title(title)
        
paths = [x[0] for x in os.walk('/home/nilesh/bladder/mri')]
imgOriginal = {}
num_count = [x.count('/') for x in paths]
lst_paths = []
lst_patient = []

for k in range(0,len(num_count)): 
    if num_count[k] == 6:
        lst_patient.append(paths[k][20:32]) 
        reader = sitk.ImageSeriesReader()
        filenamesDICOM = reader.GetGDCMSeriesFileNames(paths[k])
        reader.SetFileNames(filenamesDICOM)
        idx_patient = paths[k][20:32]+'_'+'{:02d}'.format(k)
        imgOriginal[idx_patient] = reader.Execute()
        print(idx_patient)

## 0:Slice number, 1:lower threshold, 2:higher threshold,3:index to slice,
## 4:size to slice, 5:list of seeds
## 224 for softmax
## 64 for cnn

size_width = 64
size_length = 64
position_x = 180
position_y = 180
parameters = {}
parameters['TCGA-4Z-AA80_04']=[142,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a'] 
parameters['TCGA-4Z-AA80_03']=[71,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a'] 
parameters['TCGA-4Z-AA7M_07']=[52,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T3a'] 
parameters['TCGA-4Z-AA7Y_10']=[144,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a']
parameters['TCGA-4Z-AA7Y_11']=[46,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a'] 
parameters['TCGA-ZF-AA5H_14']=[51,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2b'] 
parameters['TCGA-4Z-AA7S_17']=[122,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(60,60)],'T4a'] 
parameters['TCGA-4Z-AA81_20']=[79,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2b'] 
parameters['TCGA-4Z-AA81_21']=[17,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2b'] 
parameters['TCGA-4Z-AA86_24']=[98,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T3a'] 
parameters['TCGA-4Z-AA86_25']=[19,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T3a'] 
parameters['TCGA-4Z-AA82_28']=[32,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a'] 
parameters['TCGA-4Z-AA84_31']=[96,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T3a'] 
parameters['TCGA-4Z-AA7W_34']=[65,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(60,60)],'T2a'] 
parameters['TCGA-4Z-AA7W_35']=[535,-15,15,[position_x,position_y,0,0],[size_width,size_width,0,0],[(40,40)],'T2a']

dic_imag = {} 
dic_imag1 = {}
for pat_num in parameters.keys(): 
    print(pat_num)
    
lin_negative=60 
lin_positive=40
si_x = parameters[pat_num][3][0] 
si_y = parameters[pat_num][3][1] 
ss_x = parameters[pat_num][4][0] 
ss_y = parameters[pat_num][4][1]

for l in range(0,30): 
    idxSlice=parameters[pat_num][0] - (15-l)
    # Smoothing
    imgOriginal_sl = imgOriginal[pat_num][:,:,idxSlice] 
    dic_imag1[pat_num] = imgOriginal
    imgSmooth = sitk.CurvatureFlow(image1=imgOriginal_sl,timeStep=0.125, numberOfIterations=5)
    imgWhiteMatter2 = sitk.Shrink(imgSmooth,[8,8]) 
    y_size = imgWhiteMatter2.GetSize()[1]
    x_size = imgWhiteMatter2.GetSize()[0]
    lst = []
    for j in range(0,y_size): 
        lst_line = []
        for i in range(0,x_size): 
            lst_line.append(imgWhiteMatter2.GetPixel(i,j))
        lst.append(lst_line)
        
    idxSlice = parameters[pat_num][0]
    imgOriginal_sl = imgOriginal[pat_num][:,:,idxSlice]
    imgSmooth = sitk.CurvatureFlow(image1=imgOriginal_sl,timeStep=0.125, numberOfIterations=5)
    si_x = parameters[pat_num][3][0]
    si_y = parameters[pat_num][3][1] 
    ss_x = parameters[pat_num][4][0] 
    ss_y = parameters[pat_num][4][1]
    ## Select the bladder
    lstSeeds = parameters[pat_num][5] 
    lstSeeds1 = (40,40)
    lstSeeds2 = (40,40)
    labelWhiteMatter = 1
    labelGrayMatter = 2
    labelOtherMatter = 3
    v_lower = parameters[pat_num][1] 
    v_upper = parameters[pat_num][2]
    imgWhiteMatter6 = sitk.Threshold(image1=imgWhiteMatter2,lower=v_lower,upper=v_upper,outsideValue=0)
    name_element = pat_num + '_'+'{:02d}'.format(l)
    dic_imag[name_element] = imgWhiteMatter6
    


## Build the vector to tensor flow 
from PIL import Image
x_lst_arr = [] 
x_arr = [] 
y_lst_arr = [] 
y_arr = []

for num_img in dic_imag.keys(): 
    img = dic_imag[num_img]
    for i in range(img.GetHeight()):
        for j in range(img.GetWidth()): 
            x_arr.append(img.GetPixel(i,j))
    x_lst_arr.append(x_arr) 
    x_arr = []
    
    if parameters[num_img[0:15]][6] == 'T2a': 
        y_arr = [1,0,0,0]
    elif parameters[num_img[0:15]][6] == 'T2b':
        y_arr = [0,1,0,0]
    elif parameters[num_img[0:15]][6] == 'T3a': 
        y_arr = [0,0,1,0]
    elif parameters[num_img[0:15]][6] == 'T4a': 
        y_arr = [0,0,0,1]
    y_lst_arr.append(y_arr) 
    y_arr = []
    
size_x = len(x_lst_arr) 
size_y = len(y_lst_arr)
size_train_x = round(size_x * 2/3) 
size_test_x = size_x - size_train_x 
size_train_y = round(size_y * 2/3) 
size_test_y = size_y - size_train_y
batch_xs = np.array(x_lst_arr[0:size_train_x]) 
batch_ys = np.array(y_lst_arr[0:size_train_y]) 
batch_x_test = np.array(x_lst_arr[size_train_x:size_x]) 
batch_y_test = np.array(y_lst_arr[size_train_y:size_y])   
    
    
import tensorflow as tf
#Setup the model size_length=64 size_width=64
dim = size_length * size_width
classes = 4
x = tf.placeholder(tf.float32, [None, dim]) 
W = tf.Variable(tf.zeros([dim, classes]))
b = tf.Variable(tf.zeros([classes]))
y = tf.nn.softmax(tf.matmul(x, W) + b) 
y_ = tf.placeholder(tf.float32, [None, classes])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables() 
session_token = tf.session_tokenion() 
session_token.run(init)
session_token.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session_token.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test}))

def plot_w(weigths):
    blue_patch = mpatches.Patch(color='blue', label='T2a') 
    green_patch = mpatches.Patch(color='green', label='T2b')
    red_patch = mpatches.Patch(color='red', label='T3a') 
    gray_patch = mpatches.Patch(color='gray', label='T4a')
    plt.plot(weigths[:,0],label=blue_patch) 
    plt.plot(weigths[:,1],label=green_patch)
    plt.plot(weigths[:,2],label=red_patch)
    plt.plot(weigths[:,3],label=gray_patch,color='gray')
    handles=[red_patch,green_patch,blue_patch,gray_patch] 
    plt.ylabel('weights')
    plt.xlabel('nodes')
    plt.legend(handles=handles)
    plt.show() 
    return
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) 
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
dim = size_length * size_width
W_conv1 = weight_variable([5, 5, 1, 32]) 
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,size_width,size_length,1]) 
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_conv3 = weight_variable([5, 5, 64, 128]) 
b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


W_conv4 = weight_variable([5, 5, 128, 256]) 
b_conv4 = bias_variable([256])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)
h_pool4_flat = tf.reshape(h_pool4, [-1, 4*4*256]) 
W_fc1 = weight_variable([4*4*256, 2048])
b_fc1 = bias_variable([2048])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32) 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([2048, classes]) 
b_fc2 = bias_variable([classes])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.initialize_all_variables() 
session_token = tf.session_tokenion() 
session_token.run(init)


for i in range(100): 
    if i%10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0},session_tokenion=session_tokenion_token) 
        print("Step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5},session_tokenion=session_token)
    
print("Test accuracy %g" %accuracy.eval(feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.0},session_tokenion=session_token))