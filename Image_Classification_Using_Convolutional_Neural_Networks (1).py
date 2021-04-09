#!/usr/bin/env python
# coding: utf-8

# #**Image Classification using Convolutional Neural Networks**

# In[ ]:


__author__ = "Mohamad Radwan"
__email__ = "mohrad96@hotmail.com"


# In[ ]:


#Install Objax
get_ipython().system('pip --quiet install  objax')
import objax


# In[ ]:


import tensorflow as tf 
import tensorflow_datasets as tfds
import numpy as np
import jax.numpy as jn
import random 
import matplotlib.pyplot as plt


# ##**Part 1. Building a CNN** 
# 
# Before we build our CNN model, let's first import a dataset. For our experiment, we load the CIFAR10 dataset from Tensorflow's dataset repository. The CIFAR10 dataset consists of 60,000 32x32 colour images in 10 classes, with 6000 images per class. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.
# 
# After loading the dataset, we split the dataset into training, validation and test set. The dataset is originally stored as 50,000 training examples and 10,000 test examples. Instead, we will combine them together and make our own split.

# In[ ]:


#.load_data() by default returns a split between training and test set. 
# We then adjust the training set into a format that can be accepted by our CNN
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train.transpose(0, 3, 1, 2) / 255.0
Y_train = Y_train.flatten()
X_test = X_test.transpose(0, 3, 1, 2) / 255.0
Y_test = Y_test.flatten()

np.random.seed(1)
# To create a validation set, we first concate the original splitted dataset into a single dataset 
# then randomly shuffle the images and labels in the same way (seed = 1)
X_data = np.concatenate([X_train, X_test], axis = 0)
Y_data = np.concatenate([Y_train, Y_test], axis = 0)

N = np.arange(len(X_data))
np.random.shuffle(N)
X_data = X_data[N]
Y_data = Y_data[N]

#Next, we partition the randomly shuffled dataset into training, validation and testset according a ratio
train_ratio = 0.80
valid_ratio = 0.1
n_train = int(len(X_data) * train_ratio)
n_valid = int(len(X_data) * valid_ratio)

X_train, X_valid, X_test = X_data[:n_train], X_data[n_train:n_train+n_valid], X_data[n_train+n_valid:]
Y_train, Y_valid, Y_test = Y_data[:n_train], Y_data[n_train:n_train+n_valid], Y_data[n_train+n_valid:]


# In[ ]:


X_train.shape


# 
# Next we will construct a **Base Model**, which in our case is a small CNN.

# In[ ]:


class ConvNet(objax.Module):
  def __init__(self, number_of_channels = 3, number_of_classes = 10):
    self.conv_1 = objax.nn.Sequential([objax.nn.Conv2D(number_of_channels, 16, 2), objax.functional.relu])
    self.conv_2 = objax.nn.Sequential([objax.nn.Conv2D(16, 32, 2), objax.functional.relu])
    self.linear = objax.nn.Linear(32, number_of_classes)

  def __call__(self, x):
    x = objax.functional.max_pool_2d(self.conv_1(x), 2, 2)
    x = self.conv_2(x)
  
    x = x.mean((2,3)) #<--- global average pooling 
    x = self.linear(x)
    return x

#The following line creates the CNN
model = ConvNet()


# Before we train our conv net, let's try to better understand concepts of convolution filter and linear layer. In the following, we take the first very image of the training set, create a simple convolution routine, and show that our own routine matches what Objax returns. 
# 
# 

# In[ ]:


#Let's plot the first image in the training set.
plt.imshow(X_train[0].transpose(1,2,0))


# Next, we will pass our image through Objax's convolution routine.

# In[ ]:


# We append the first image with a batch size of 1 so it can be fed into a convolution layer
my_image = np.expand_dims(X_train[0], 0)
print(my_image.shape)
#Consider a very simple CNN filter with stride = 1 and no padding ('VALID').
Conv2d = objax.nn.Conv2D(nin = 3, nout = 2, k = 1, strides = 1, padding = 'VALID', use_bias = False)

filter_weights = Conv2d.w.value #This is the initial weight of the filter, which we gradually update when training, we ignore bias for now
print("Filter weights:", filter_weights)
print("Conv output:", Conv2d(my_image))
Convoluted_Image = Conv2d(my_image)
print("Conv output shape:", np.shape(Conv2d(my_image)))


# **In the cells below, we create our own convolution routine that takes in the image and the initial weights used by Objax's own convolution routine (Conv2d.w.value) and then show that our convolution routine returns the same value as Objax's.**

# In[ ]:


def my_conv_net(my_image, initial_filter_weights):
  Hout = my_image.shape[2]-initial_filter_weights.shape[0]+1
  Wout = my_image.shape[3]-initial_filter_weights.shape[0]+1
  weights = initial_filter_weights
  my_conv_output = np.zeros((my_image.shape[0],initial_filter_weights.shape[3],Hout,Wout))
  for n in range(my_image.shape[0]):
    for c in range(initial_filter_weights.shape[3]):
      for h in range(Hout):
        for w in range(Wout):
          my_conv_output[n,c,h,w] = jn.dot(weights[:,:,:,c], my_image[n,:,h,w])
  
  return my_conv_output


# In[ ]:


my_conv_output = my_conv_net(my_image, filter_weights)


# In[ ]:


print((Convoluted_Image.shape == my_conv_output.shape))
print((Convoluted_Image == my_conv_output))


# In[ ]:


print(my_conv_output)


# Comparing my convoluted output and objax's convoluted output we can see that they are equal, there are minor differences possibly due to differences in rounding and approximations of the calculations between objax's convolution routine and mine.

# The outputs of last convolution layer is typically rearranged so it can be fed into a linear layer. Calling .mean((2,3)) rearranges the output of our convolution routine.

# In[ ]:


#Check that .mean((2,3)) rearranges the image
my_conv_output.mean((2,3))
my_conv_output.mean((2,3)).shape


# Take our rearranged output and feed it into a linear layer of appropriate size. 
# 
# **Implementing the linear layer using one line of code and showing that it provides the same value as Objax's own linear layer.** 
# 
# 

# In[ ]:


#Objax
Linear_Layer = objax.nn.Linear(my_conv_output.mean((2,3)).shape[1], 1)
Y = Linear_Layer(my_conv_output.mean((2,3)))
w = Linear_Layer.w.value
b = Linear_Layer.b.value
print(Y)


# In[ ]:


#Manually
Y_manual = jn.dot(my_conv_output.mean((2,3)), w) + b
Y_manual


# Objax's Linear layer output and my manual linear layer output are equal.

# ##**Part 2. Training and Tuning a CNN**
# 
# The following starter code trains the neural network in Part 1. 

# In[ ]:


#Define loss function as averaged value of of cross entropies
def loss_function(x, labels):
    logit = model(x)
    return objax.functional.loss.cross_entropy_logits_sparse(logit, labels).mean()

#Define a prediction function
predict = objax.Jit(lambda x: objax.functional.softmax(model(x)), model.vars()) 

#Create an object that can be used to calculate the gradient and value of loss_function
gv= objax.GradValues(loss_function, model.vars())

#Create an object that can be used to provide trainable variables in the model
tv = objax.ModuleList(objax.TrainRef(x) for x in model.vars().subset(objax.TrainVar))

#Training routine
def train_op(x, y, learning_rate):
    lr = learning_rate
    gradient, loss_value = gv(x, y)   # calculate gradient and loss value "backprop"
    #next we update the trainable parameter using SGD and similar procedure
    for grad, params in zip(gradient, tv.vars()):
      params.value = params.value - lr*grad

    return loss_value                      # return loss value

#make train_op (much) faster using JIT compilation
train_op = objax.Jit(train_op, gv.vars() + tv.vars())


# In[ ]:


def train(EPOCHS = 20, BATCH = 32, LEARNING_RATE = 9e-4):
  avg_train_loss_epoch = []
  avg_val_loss_epoch = []
  train_acc_epoch = []
  val_acc_epoch = []

  for epoch in range(EPOCHS):
      avg_train_loss = 0 # (averaged) training loss per batch
      avg_val_loss =  0  # (averaged) validation loss per batch
      train_acc = 0      # training accuracy per batch
      val_acc = 0        # validation accuracy per batch

      # shuffle the examples prior to training to remove correlation 
      train_indices = np.arange(len(X_train)) 
      np.random.shuffle(train_indices)
      for it in range(0, X_train.shape[0], BATCH):
          batch = train_indices[it:(it+np.minimum(BATCH,X_train.shape[0]-it))]
          avg_train_loss += float(train_op(X_train[batch], Y_train[batch], LEARNING_RATE)[0]) * len(batch)
          train_prediction = predict(X_train[batch]).argmax(1)
          train_acc += (np.array(train_prediction).flatten() == Y_train[batch]).sum()
      train_acc_epoch.append(100*train_acc/X_train.shape[0])
      avg_train_loss_epoch.append(avg_train_loss/X_train.shape[0])

      # run validation
      val_indices = np.arange(len(X_valid)) 
      np.random.shuffle(val_indices)    
      for it in range(0, X_valid.shape[0], BATCH):
          batch = val_indices[it:(it+np.minimum(BATCH,X_valid.shape[0]-it))]
          avg_val_loss += float(loss_function(X_valid[batch], Y_valid[batch])) * len(batch)
          val_prediction = predict(X_valid[batch]).argmax(1)
          val_acc += (np.array(val_prediction).flatten() == Y_valid[batch]).sum()
      val_acc_epoch.append(100*val_acc/X_valid.shape[0])
      avg_val_loss_epoch.append(avg_val_loss/X_valid.shape[0])

      print('Epoch %04d  Training Loss %.2f Validation Loss %.2f Training Accuracy %.2f Validation Accuracy %.2f' % (epoch + 1, avg_train_loss/X_train.shape[0], avg_val_loss/X_valid.shape[0], 100*train_acc/X_train.shape[0], 100*val_acc/X_valid.shape[0]))


  print("Base model: Max Validation Accuracy: ",round(max(val_acc_epoch),3),"% at epoch ", val_acc_epoch.index(max(val_acc_epoch))+1)
  #Plot training loss
  plt.title("Train vs Validation Loss")
  plt.plot(avg_train_loss_epoch, label="Train")
  plt.plot(avg_val_loss_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()

  plt.title("Train vs Validation Accuracy")
  plt.plot(train_acc_epoch, label="Train")
  plt.plot(val_acc_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.legend(loc='best')
  plt.show()


# In[ ]:


train(EPOCHS = 60)


# **Next, we optimize the base model by including 4 tunable hyperparamters which are BATCH, LEARNING_RATE, Nb_Outputs_ConvolutionLayer1, Nb_Convolution_Layers.**

# In[ ]:


def train_4Hyperparameters(EPOCHS, BATCH, LEARNING_RATE, Nb_Outputs_ConvolutionLayer1, Nb_Convolution_Layers, ModelName, X_train, X_valid, X_test, Y_train, Y_valid, Y_test):
  class ConvNetUpdated(objax.Module):
    def __init__(self, Nb_Outputs_ConvolutionLayer1 = Nb_Outputs_ConvolutionLayer1, Nb_Convolution_Layers = Nb_Convolution_Layers, number_of_channels = 3, number_of_classes = 10):
      self.conv_1 = objax.nn.Sequential([objax.nn.Conv2D(number_of_channels, Nb_Outputs_ConvolutionLayer1, 2), objax.functional.relu])
      n = 1
      self.conv = []
      for i in range(Nb_Convolution_Layers - 1):
        self.conv.append(objax.nn.Sequential([objax.nn.Conv2D(n*Nb_Outputs_ConvolutionLayer1, 2*n*Nb_Outputs_ConvolutionLayer1, 2), objax.functional.relu]))
        n = n*2
      self.linear = objax.nn.Linear(n*Nb_Outputs_ConvolutionLayer1, number_of_classes)
     

    def __call__(self, x):
      if Nb_Convolution_Layers == 1:
        x = self.conv_1(x)
        x = x.mean((2,3)) #<--- global average pooling 
        x = self.linear(x)
      elif Nb_Convolution_Layers > 1:
        x = objax.functional.max_pool_2d(self.conv_1(x), 2, 2)
        for i in range(Nb_Convolution_Layers - 1):
          if (i < Nb_Convolution_Layers - 2):
            x = objax.functional.max_pool_2d(self.conv[i](x), 2, 2)
          else:
            x = self.conv[i](x)
            x = x.mean((2,3)) #<--- global average pooling 
            x = self.linear(x)
      return x

  #The following line creates the CNN
  model = ConvNetUpdated()

  def loss_function(x, labels):
    logit = model(x)
    return objax.functional.loss.cross_entropy_logits_sparse(logit, labels).mean()

  #Define a prediction function
  predict = objax.Jit(lambda x: objax.functional.softmax(model(x)), model.vars()) 

  #Create an object that can be used to calculate the gradient and value of loss_function
  gv= objax.GradValues(loss_function, model.vars())

  #Create an object that can be used to provide trainable variables in the model
  tv = objax.ModuleList(objax.TrainRef(x) for x in model.vars().subset(objax.TrainVar))

  #Training routine
  def train_op(x, y, learning_rate):
      lr = learning_rate
      gradient, loss_value = gv(x, y)   # calculate gradient and loss value "backprop"
      #next we update the trainable parameter using SGD and similar procedure
      for grad, params in zip(gradient, tv.vars()):
        params.value = params.value - lr*grad

      return loss_value                      # return loss value

  #make train_op (much) faster using JIT compilation
  train_op = objax.Jit(train_op, gv.vars() + tv.vars())


  avg_train_loss_epoch = []
  avg_val_loss_epoch = []
  train_acc_epoch = []
  val_acc_epoch = []

  for epoch in range(EPOCHS):
      avg_train_loss = 0 # (averaged) training loss per batch
      avg_val_loss =  0  # (averaged) validation loss per batch
      train_acc = 0      # training accuracy per batch
      val_acc = 0        # validation accuracy per batch

      # shuffle the examples prior to training to remove correlation 
      train_indices = np.arange(len(X_train)) 
      np.random.shuffle(train_indices)
      for it in range(0, X_train.shape[0], BATCH):
          batch = train_indices[it:(it+np.minimum(BATCH,X_train.shape[0]-it))]
          avg_train_loss += float(train_op(X_train[batch], Y_train[batch], LEARNING_RATE)[0]) * len(batch)
          train_prediction = predict(X_train[batch]).argmax(1)
          train_acc += (np.array(train_prediction).flatten() == Y_train[batch]).sum()
      train_acc_epoch.append(100*train_acc/X_train.shape[0])
      avg_train_loss_epoch.append(avg_train_loss/X_train.shape[0])

      # run validation
      val_indices = np.arange(len(X_valid)) 
      np.random.shuffle(val_indices)    
      for it in range(0, X_valid.shape[0], BATCH):
          batch = val_indices[it:(it+np.minimum(BATCH,X_valid.shape[0]-it))]
          avg_val_loss += float(loss_function(X_valid[batch], Y_valid[batch])) * len(batch)
          val_prediction = predict(X_valid[batch]).argmax(1)
          val_acc += (np.array(val_prediction).flatten() == Y_valid[batch]).sum()
      val_acc_epoch.append(100*val_acc/X_valid.shape[0])
      avg_val_loss_epoch.append(avg_val_loss/X_valid.shape[0])
      
      print('Epoch %04d  Training Loss %.2f Validation Loss %.2f Training Accuracy %.2f Validation Accuracy %.2f' % (epoch + 1, avg_train_loss/X_train.shape[0], avg_val_loss/X_valid.shape[0], 100*train_acc/X_train.shape[0], 100*val_acc/X_valid.shape[0]))


  #Printing the epoch that has maximum validation accuracy
  print(ModelName+": Max Validation Accuracy: ",round(max(val_acc_epoch),3),"% at epoch ", val_acc_epoch.index(max(val_acc_epoch))+1)

  #Plot training loss
  plt.title("Train vs Validation Loss")
  plt.plot(avg_train_loss_epoch, label="Train")
  plt.plot(avg_val_loss_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()

  plt.title("Train vs Validation Accuracy")
  plt.plot(train_acc_epoch, label="Train")
  plt.plot(val_acc_epoch, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Accuracy (%)")
  plt.legend(loc='best')
  plt.show()
  
  #Storing Test Accuracy
  test_prediction = predict(X_test).argmax(1)
  test_acc = 100*((np.array(test_prediction).flatten() == Y_test).sum())/X_test.shape[0]
  return round(test_acc,3)


# Trying 2 different models with different hyperparamter sets.

# In[ ]:


M1_Testing_Accuracy = train_4Hyperparameters(EPOCHS = 60, BATCH = 24, LEARNING_RATE = 0.03, Nb_Outputs_ConvolutionLayer1 = 16, Nb_Convolution_Layers = 2, 
                                             ModelName = 'M1', X_train = X_train, X_valid = X_valid, X_test = X_test, Y_train = Y_train, Y_valid = Y_valid, Y_test = Y_test)


# In[ ]:


M2_Testing_Accuracy = train_4Hyperparameters(EPOCHS = 60, BATCH = 40, LEARNING_RATE = 0.01, Nb_Outputs_ConvolutionLayer1 = 20, Nb_Convolution_Layers = 3, 
                                             ModelName = 'M2', X_train = X_train, X_valid = X_valid, X_test = X_test, Y_train = Y_train, Y_valid = Y_valid, Y_test = Y_test)


# In[ ]:


print("Testing Accuracy:", M1_Testing_Accuracy,'%')


# ##**Part 3. Trying Out a New Dataset**
# 

# Next we try out a new dataset called "cmaterdb". This dataset contains images of - Handwritten Bangla numerals - balanced dataset of total 6000 Bangla numerals (32x32 RGB coloured, 6000 images), each having 600 images per class(per digit). Handwritten Devanagari numerals - balanced dataset of total 3000 Devanagari numerals (32x32 RGB coloured, 3000 images), each having 300 images per class(per digit). Handwritten Telugu numerals - balanced dataset of total 3000 Telugu numerals (32x32 RGB coloured, 3000 images), each having 300 images per class(per digit).

# In[ ]:


numerals_data = tfds.load('cmaterdb', as_supervised = True)
numerals_train, numerals_test = numerals_data['train'], numerals_data['test']
X_train1 = np.zeros((5000, 32, 32, 3))
Y_train1 = np.zeros((5000,1))
X_test1 = np.zeros((1000, 32, 32, 3))
Y_test1 = np.zeros((1000,1))
i=0
for image, label in numerals_train:
  X_train1[i] = image
  Y_train1[i] = label
  i = i+1
i=0
for image, label in numerals_test:
  X_test1[i] = image
  Y_test1[i] = label
  i = i+1


# In[ ]:


X_train1 = X_train1.transpose(0, 3, 1, 2) / 255.0
Y_train1 = Y_train1.flatten()
X_test1 = X_test1.transpose(0, 3, 1, 2) / 255.0
Y_test1 = Y_test1.flatten()

np.random.seed(1)
# To create a validation set, we first concate the original splitted dataset into a single dataset 
# then randomly shuffle the images and labels in the same way (seed = 1)
X_data1 = np.concatenate([X_train1, X_test1], axis = 0)
Y_data1 = np.concatenate([Y_train1, Y_test1], axis = 0)

N1 = np.arange(len(X_data1))
np.random.shuffle(N1)
X_data1 = X_data1[N1]
Y_data1 = Y_data1[N1]

#Next, we partition the randomly shuffled dataset into training, validation and testset according a ratio
train_ratio1 = 0.80
valid_ratio1 = 0.1
n_train1 = int(len(X_data1) * train_ratio1)
n_valid1 = int(len(X_data1) * valid_ratio1)

X_train1, X_valid1, X_test1 = X_data1[:n_train1], X_data1[n_train1:n_train1+n_valid1], X_data1[n_train1+n_valid1:]
Y_train1, Y_valid1, Y_test1 = Y_data1[:n_train1], Y_data1[n_train1:n_train1+n_valid1], Y_data1[n_train1+n_valid1:]


# In[ ]:


Testing_Accuracy_Base_Model = train_4Hyperparameters(EPOCHS = 150, BATCH = 10, LEARNING_RATE = 1e-3, Nb_Outputs_ConvolutionLayer1 = 16, Nb_Convolution_Layers = 2, 
                                                     ModelName = "BaseModel", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


Testing_Accuracy_M = train_4Hyperparameters(EPOCHS = 150, BATCH = 10, LEARNING_RATE = 1e-2, Nb_Outputs_ConvolutionLayer1 = 16, Nb_Convolution_Layers = 3, 
                                                     ModelName = "M", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


print("Testing Accuracy:", Testing_Accuracy_M,'%')


# ##**Part 4. Open-Ended Exploration**

# Question: How do Hyperparamters interact?

# In[ ]:


Testing_Accuracy_Base = train_4Hyperparameters(EPOCHS = 150, BATCH = 32, LEARNING_RATE = 5e-3, Nb_Outputs_ConvolutionLayer1 = 16, Nb_Convolution_Layers = 1, 
                                                     ModelName = "Base", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


#Batch
M1_Testing_Accuracy = train_4Hyperparameters(EPOCHS = 150, BATCH = 16, LEARNING_RATE = 5e-3, Nb_Outputs_ConvolutionLayer1 = 16, Nb_Convolution_Layers = 1, 
                                                     ModelName = "M1", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


#Learning Rate
M2_Testing_Accuracy = train_4Hyperparameters(EPOCHS = 150, BATCH = 32, LEARNING_RATE = 5e-2, Nb_Outputs_ConvolutionLayer1 = 16, Nb_Convolution_Layers = 1, 
                                                     ModelName = "M2", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


#Batch and Learning rate
M3_Testing_Accuracy = train_4Hyperparameters(EPOCHS = 150, BATCH = 16, LEARNING_RATE = 5e-2, Nb_Outputs_ConvolutionLayer1 = 16, Nb_Convolution_Layers = 1, 
                                                     ModelName = "M3", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


#Base2
Testing_Accuracy_Base2 = train_4Hyperparameters(EPOCHS = 150, BATCH = 16, LEARNING_RATE = 5e-2, Nb_Outputs_ConvolutionLayer1 = 128, Nb_Convolution_Layers = 2, 
                                                     ModelName = "Base2", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


#Number of outputs of convolution layer 1
M4_Testing_Accuracy = train_4Hyperparameters(EPOCHS = 150, BATCH = 16, LEARNING_RATE = 5e-2, Nb_Outputs_ConvolutionLayer1 = 256, Nb_Convolution_Layers = 2, 
                                                     ModelName = "M4", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


#Number of convolution layers
M5_Testing_Accuracy = train_4Hyperparameters(EPOCHS = 150, BATCH = 16, LEARNING_RATE = 5e-2, Nb_Outputs_ConvolutionLayer1 = 128, Nb_Convolution_Layers = 3, 
                                                     ModelName = "M5", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


#Number of outputs of convolution layer 1 and Number of convolution layers
M6_Testing_Accuracy = train_4Hyperparameters(EPOCHS = 150, BATCH = 16, LEARNING_RATE = 5e-2, Nb_Outputs_ConvolutionLayer1 = 256, Nb_Convolution_Layers = 3, 
                                                     ModelName = "M6", X_train = X_train1, X_valid = X_valid1, X_test = X_test1, Y_train = Y_train1.astype(int), Y_valid = Y_valid1.astype(int), Y_test = Y_test1.astype(int))


# In[ ]:


print("Testing Accuracy:", M6_Testing_Accuracy,'%')

