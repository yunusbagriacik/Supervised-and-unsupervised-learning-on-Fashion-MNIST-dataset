#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.io import imread, imshow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torchvision
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import losses
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Input, MaxPooling2D, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import visualkeras
import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# ## B2. Supervised learning on Fashion-MNIST 

# ### B1. Reproducibility & readability 
# Whenever there is randomness in the computation, you MUST set a random seed for reproducibility. Use your UCard number XXXXXXXXX as the random seed throughout this assignment. [1 mark]
# 

# In[ ]:


RANDOM_STATE = 2747996


# ### B2.1 Data loading and inspection 

# a)	Use the PyTorch API for Fashion-MNIST to load both the training and test data of Fashion-MNIST. You may refer to similar procedures in Lab 7 for CIFAR-10. Preprocessing is NOT required but you are encouraged to explore and use preprocessing such as those in the torchvision.transforms API. [1 mark]

# In[2]:


# get the current working directory and add data to that folder, Pytorch will automatically create the data folder.
cwd = os.getcwd() + '/data'


# In[3]:


# get if not present already download the data
train_set = torchvision.datasets.FashionMNIST(cwd, download=True, transform=
                                                transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST(cwd, download=True, train=False, transform=
                                               transforms.Compose([transforms.ToTensor()])) 


# In[ ]:


train_set


# In[ ]:


test_set


# In[ ]:


train = train_set.train_data
test = test_set.test_data


# In[ ]:


X_train = np.array(train)
y_train = np.array(train_set.train_labels)
X_test = np.array(test)
y_test = np.array(test_set.test_labels)


# b)	Display at least eight images for each of the 10 classes (8x10=80 images). [1 mark]

# In[ ]:


class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
i = 0
count = 0
# loop through each class
for j in range(10):
    # get first 10 images of current class, and loop through it
    for i in np.where(y_train == j)[0][:10]:
        plt.subplot(10, 10, count + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i]) 
        count += 1
        label_index = int(y_train[i])
        plt.title(class_names[label_index])
plt.show()


# ### B2.2 Evaluation metrics

# M1) Training accuracy: the prediction accuracy of the trained model on the training dataset. <br>
# M2) Testing accuracy: the prediction accuracy of the trained model on the test dataset. <br>
# M3) Training time: the time taken to train the model (i.e. to learn/estimate the learnable parameters) on the training dataset. <br>
# M4) The number of learnable parameters of the model.
# 

# ### B2.3 Logistic regression

# a)	Train a logistic regression model on the training set of Fashion-MNIST and test the trained model on the test set of Fashion-MNIST. Report the four metrics M1 to M 4 and plot a confusion matrix for predictions on the test data. [2 marks]

# In[ ]:


X_train = np.array(train) / 255
X_test = np.array(test) / 255
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)


# ### Logistic Regression

# In[ ]:


start = time.time()
clf = LogisticRegression(random_state=RANDOM_STATE).fit(X_train, y_train)
stop = time.time()
training_time = stop - start
print(f"Calculation time: {training_time}s")


# In[ ]:


testing_score = clf.score(X_test, y_test) * 100
training_score = clf.score(X_train, y_train) * 100
print('Training Accuracy:', training_score)
print('Testing Accuracy:', testing_score)


# In[ ]:


y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True)
plt.title('Logisitic Regression Training Confusion Matrix')
plt.show()


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True)
plt.title('Logisitic Regression Testing Confusion Matrix')
plt.show()


# In[ ]:


results = []
results.append({'Classifier Name': 'Logistic Regression', 'Training Accuracy': training_score,
                'Testing Accuracy': testing_score, 'Training Time': training_time, 'Learnable Parameters': len(clf.get_params())})


# b)	Train and test a logistic regression model with L1 regularisation as in a). Report M1 to M4 and plot a confusion matrix for predictions on the test data [1 mark]

# ### Logistic Regression with L1

# In[ ]:


start = time.time()
clf = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
stop = time.time()
training_time = stop - start
print(f"Calculation time: {training_time}s")


# In[ ]:


testing_score = clf.score(X_test, y_test) * 100
training_score = clf.score(X_train, y_train) * 100
print('Training Accuracy:', training_score)
print('Testing Accuracy:', testing_score)


# In[ ]:


y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True)
plt.title('Logisitic Regression with L1 Training Confusion Matrix')
plt.show()


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True)
plt.title('Logisitic Regression with L1 Testing Confusion Matrix')
plt.show()


# In[ ]:


results.append({'Classifier Name': 'Logistic Regression with L1', 'Training Accuracy': training_score,
                'Testing Accuracy': testing_score, 'Training Time': training_time, 'Learnable Parameters': len(clf.get_params())})


# c)	Train and test a logistic regression model with L2 regularisation as in a). Report M1 to M4 and plot a confusion matrix for predictions on the test data [1 mark]

# ### Logisitic Regression with L2

# In[ ]:


start = time.time()
clf = LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', random_state=RANDOM_STATE, n_jobs=-1).fit(X_train, y_train)
stop = time.time()
training_time = stop - start
print(f"Calculation time: {training_time}s")


# In[ ]:


testing_score = clf.score(X_test, y_test) * 100
training_score = clf.score(X_train, y_train) * 100
print('Training Accuracy:', training_score)
print('Testing Accuracy:', testing_score)


# In[ ]:


y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True)
plt.title('Logisitic Regression with L2 Training Confusion Matrix')
plt.show()


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True)
plt.title('Logisitic Regression with L2 Testing Confusion Matrix')
plt.show()


# In[ ]:


results.append({'Classifier Name': 'Logistic Regression with L2', 'Training Accuracy': training_score,
                'Testing Accuracy': testing_score, 'Training Time': training_time, 'Learnable Parameters': len(clf.get_params())})


# ### B2.4 Convolutional Neural networks 

# a)	Design a CNN with two Conv layers and two FC layers. Train and test it as in B2.3a. Report M1 to M4 and plot a confusion matrix for predictions on the test data. [2 marks]

# ### CNN with 2Conv+2Dense Layers

# In[4]:


X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)


# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(10, activation='softmax'))

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


from keras.utils.layer_utils import count_params


# In[ ]:


visualkeras.layered_view(model)


# In[ ]:


y_cat_train = to_categorical(y_train, num_classes= 10)
y_cat_test = to_categorical(y_test, num_classes= 10)


# In[ ]:


early_stop = EarlyStopping(monitor = 'val_loss', patience = 3)


# In[ ]:


start = time.time()
model.fit(x = X_train, y = y_cat_train, batch_size = 128, epochs = 100, validation_data = (X_test, y_cat_test), callbacks = [early_stop])
stop = time.time()
training_time = stop - start


# In[ ]:


test_predictions = model.predict(X_test)
y_pred_test = np.argmax(test_predictions, axis = 1)
y_test = np.argmax(y_cat_test, axis = 1)

train_predictions = model.predict(X_train)
y_pred_train = np.argmax(train_predictions, axis = 1)
y_train = np.argmax(y_cat_train, axis = 1)


# In[ ]:


testing_score = accuracy_score(y_test, y_pred_test) * 100
training_score = accuracy_score(y_train, y_pred_train) * 100
print('Training Accuracy:', training_score)
print('Testing Accuracy:', testing_score)


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True)
plt.title('CNN 2Conv+2Dense Layers Training Confusion Matrix')
plt.show()


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True)
plt.title('CNN 2Conv+2Dense Layers Testing Confusion Matrix')
plt.show()


# In[ ]:


results.append({'Classifier Name': 'CNN 2Conv+2Dense', 'Training Accuracy': training_score,
                'Testing Accuracy': testing_score, 'Training Time': training_time, 'Learnable Parameters': count_params(model.trainable_weights)})


# b)	Design a CNN with two Conv layers and five FC layers. Train and test it as in B2.3a. Report M1 to M4 and plot a confusion matrix for predictions on the test data. [2 marks]

# ### CNN with 2Conv+5Dense Layers

# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.3))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(10, activation='softmax'))

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


visualkeras.layered_view(model)


# In[ ]:


start = time.time()
model.fit(x = X_train, y = y_cat_train, batch_size = 128, epochs = 100, validation_data = (X_test, y_cat_test), callbacks = [early_stop])
stop = time.time()
training_time = stop - start


# In[ ]:


test_predictions = model.predict(X_test)
y_pred_test = np.argmax(test_predictions, axis = 1)
y_test = np.argmax(y_cat_test, axis = 1)

train_predictions = model.predict(X_train)
y_pred_train = np.argmax(train_predictions, axis = 1)
y_train = np.argmax(y_cat_train, axis = 1)


# In[ ]:


testing_score = accuracy_score(y_test, y_pred_test) * 100
training_score = accuracy_score(y_train, y_pred_train) * 100
print('Training Accuracy:', training_score)
print('Testing Accuracy:', testing_score)


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True)
plt.title('CNN 2Conv+5Dense Layers Training Confusion Matrix')
plt.show()


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True)
plt.title('CNN 2Conv+5Dense Layers Testing Confusion Matrix')
plt.show()


# In[ ]:


results.append({'Classifier Name': 'CNN 2Conv+5Dense', 'Training Accuracy': training_score,
                'Testing Accuracy': testing_score, 'Training Time': training_time, 'Learnable Parameters': count_params(model.trainable_weights)})


# c)	Design a CNN with five Conv layers and two FC layers. Train and test it as in B2.3a. Report M1 to M4 and plot a confusion matrix for predictions on the test data. [2 marks]

# ### CNN with 5Conv+2Dense Layers

# In[ ]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.3))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(Conv2D(filters = 128, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(Conv2D(filters = 128, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(Dropout(rate = 0.3))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(10, activation='softmax'))

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


visualkeras.layered_view(model)


# In[ ]:


start = time.time()
model.fit(x = X_train, y = y_cat_train, batch_size = 128, epochs = 100, validation_data = (X_test, y_cat_test), callbacks = [early_stop])
stop = time.time()
training_time = stop - start


# In[ ]:


test_predictions = model.predict(X_test)
y_pred_test = np.argmax(test_predictions, axis = 1)
y_test = np.argmax(y_cat_test, axis = 1)

train_predictions = model.predict(X_train)
y_pred_train = np.argmax(train_predictions, axis = 1)
y_train = np.argmax(y_cat_train, axis = 1)


# In[ ]:


testing_score = accuracy_score(y_test, y_pred_test) * 100
training_score = accuracy_score(y_train, y_pred_train) * 100
print('Training Accuracy:', training_score)
print('Testing Accuracy:', testing_score)


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True)
plt.title('CNN 5Conv+2Dense Layers Training Confusion Matrix')
plt.show()


# In[ ]:


plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True)
plt.title('CNN 5Conv+2Dense Layers Testing Confusion Matrix')
plt.show()


# In[ ]:


results.append({'Classifier Name': 'CNN 5Conv+2Dense', 'Training Accuracy': training_score,
                'Testing Accuracy': testing_score, 'Training Time': training_time, 'Learnable Parameters': count_params(model.trainable_weights)})


# ### B2.4 Performance comparison 

# a)	Summarise each of the four metrics from the six models in B2.3 and B2.4 using a bar graph. In total, four bar graphs need to be generated and displayed, one for each metric with six results from B2.3 and B2.4. [1 mark]

# In[ ]:


performance_df = pd.json_normalize(results).set_index('Classifier Name')


# In[ ]:


performance_df


# In[ ]:


performance_df[['Training Accuracy', 'Testing Accuracy']].plot(kind='bar', figsize=(15, 6), grid=True)
plt.title('Training & Testing Accuracies', fontsize=16)
plt.show()


# In[ ]:


performance_df['Training Time'].plot(figsize=(15, 6), grid=True)
plt.title('Training Time in seconds', fontsize=16)
plt.show()


# In[ ]:


performance_df['Learnable Parameters'].plot(figsize=(15, 6), grid=True)
plt.title('Learnable Parameters Count', fontsize=16)
plt.show()


# b)	Describe at least two observations interesting to you. [1 mark]

# **Observations:**
# - Logistic Regression with default parameters takes much less time than those which has parameters.
# - Conv layers takes more time for calculation than the dense layers.
# - CNN took alot of time to train, but provided the better results.
# - Non of the model seems to be over-fit or under-fit.

# ## B3. Unsupervised learning on Fashion-MNIST 

# ### B3.1 PCA and k-means 

# In[ ]:


train_idx0 = np.where(y_train == 0)[0]
train_idx1 = np.where(y_train == 5)[0]
test_idx0 = np.where(y_test == 0)[0]
test_idx1 = np.where(y_test == 5)[0]


# In[ ]:


train_idx = np.concatenate([train_idx0,train_idx1])
test_idx = np.concatenate([test_idx0,test_idx1])


# In[ ]:


X_train2 = np.array(train)[train_idx] / 255
y_train2 = np.array(train_set.train_labels)[train_idx]
X_test2 = np.array(test)[test_idx] / 255
y_test2 = np.array(test_set.test_labels)[test_idx]


# In[ ]:


X_train2.shape


# In[ ]:


training_images_1d = [x.ravel() for x in X_train2]
testing_images_1d = [x.ravel() for x in X_test2]


# a)	Apply PCA to all images of these two chosen classes. Visualise the top 24 eigenvectors as images and display them in the order of descending corresponding values (the one corresponding to the largest eigenvalue first).  [2 marks]

# In[ ]:


pca = PCA(n_components=24)
pca.fit(training_images_1d)
X_train_pca = pca.transform(training_images_1d)
X_test_pca = pca.transform(testing_images_1d)


# In[ ]:


count = 1
for comp in pca.components_:
    plt.imshow(comp.reshape(28, 28),cmap='gray')
    plt.title('Component '+ str(count))
    plt.show()
    count += 1


# b)	Use the top 24 PCs to reconstruct 30 images, with 15 from each class (any 15 images are fine from each class). Compute and report the mean squared error between the reconstructed and original images for these 30 images (a single value to be reported). Show these 30 pairs of reconstructed and original images. [2 marks]

# In[ ]:


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

count = 0

mses = []

for j in [0, 5]:
    for i in np.where(y_train2 == j)[0][:15]:
        plt.figure()

        f, axarr = plt.subplots(1,2) 
        pca_image = pca.inverse_transform(X_train_pca[i]).reshape(28, 28)
        axarr[0].imshow(pca_image,cmap='gray')
        axarr[1].imshow(X_train2[i],cmap='gray')
        
        err = mse(pca_image, X_train2[i])
        mses.append(err)
        label_index = int(y_train2[i])
        axarr[0].set_title(class_names[label_index] + ' PCA Image')
        axarr[1].set_title(class_names[label_index] + ' Original Image')
        
        count += 1
        
plt.show()


# In[ ]:


print('Avg MSE:', sum(mses)/len(mses))


# c)	Plot the PCA representations of all data points in a 2D plane using the top two PCs. Use different colours/markers for the two classes for better visualisation (Hint: You need to use the class labels here for visualisation). [2 marks]

# In[ ]:


pca = PCA(n_components=2)
pca.fit(training_images_1d)
X_train_pca = pca.transform(training_images_1d)
X_test_pca = pca.transform(testing_images_1d)


# In[ ]:


X_train_df = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
X_train_df['class'] = y_train2
X_test_df = pd.DataFrame(X_test_pca, columns=['PC1', 'PC2'])
X_test_df['class'] = y_test2

X_pca_df = pd.concat([X_train_df, X_test_df])


# In[ ]:


plt.figure(figsize=(12, 6))
plt.scatter(X_pca_df['PC1'], X_pca_df['PC2'], c=X_pca_df['class'], s=50, cmap='viridis')
plt.title('PCA Representations')
plt.show()


# d)	Use k-means to cluster all data points as represented by the top two PCs (clustering of two-dimensional vectors, where each vector has two values, PC1 and PC2). Visualise the two clusters with different colours/markers and indicate the cluster centers clearly with a marker in a figure similar to question c) above. [1 mark].

# In[ ]:


X = X_pca_df[['PC1', 'PC2']]
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# In[ ]:


plt.figure(figsize=(12, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('K-Means Predicitions')
plt.show()


# ### B3.2 AutoEncoder 

# a)	Design a new autoencoder with five Conv2d layers and five ConvTranspose2d layers. You are free to choose the activation functions and settings such as stride and padding. Train this new autoencoder on all images of these two chosen classes for at least 20 epochs. Plot the mean squared error against the epoch. [2 marks]

# In[5]:


latent_dim = 64 

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   

        self.encoder = Sequential([ 
            Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(28, 28, 1)),
            Conv2D(16, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'),
            Conv2D(8, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'),
            Conv2D(4, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'),
            Conv2D(2, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'),

            
        ])
        
        self.decoder = Sequential([
            Conv2DTranspose(2, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal'),
            Conv2DTranspose(4, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal'),
            Conv2DTranspose(8, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal'),
            Conv2DTranspose(16, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal'),
            Conv2DTranspose(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal'),
            Conv2D(1, kernel_size=(3,3), activation='relu', padding='same')
        ])
        

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(latent_dim)


# In[ ]:


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


# In[ ]:


X_train3 = X_train2.reshape(X_train2.shape[0],28,28,1)
X_test3 = X_test2.reshape(X_test2.shape[0],28,28,1)

autoencoder.fit(X_train3, X_train3,epochs=20, validation_data=(X_test3, X_test3))


# In[ ]:


encoded_imgs = autoencoder.encoder(X_test2).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


# In[ ]:


mses = []
count = 0
for j in [0, 5]:
    for i in np.where(y_test2 == j)[0][:15]:
        plt.figure()

        f, axarr = plt.subplots(1,2) 
        axarr[0].imshow(decoded_imgs[i].reshape(28, 28),cmap='gray')
        axarr[1].imshow(X_test2[i],cmap='gray')
        label_index = int(y_test2[i])
        axarr[0].set_title(class_names[label_index] + ' Reconstructed Image')
        axarr[1].set_title(class_names[label_index] + ' Original Image')
        
        count += 1
        
plt.show()


# In[ ]:


metrics = pd.DataFrame(autoencoder.history.history)


# In[ ]:


metrics[['loss', 'val_loss']].plot(figsize=(12, 6), grid=True)
plt.title('Loss and Valdation Loss')
plt.xlabel('Epoch No.')
plt.xticks(range(0,20))

plt.show()


# b)	Modify the autoencoder in 3.2a so that the code (bottleneck) has a dimension of 2 only. Plot the 2-dimensional representations in terms of this autoencoder code for all data points in a 2D plane as in 3.1c and cluster them as in 3.1d, showing similar colour/marker visualisation. [2 marks]

# In[ ]:


latent_dim = 2

class Autoencoder2D(Model):
    def __init__(self, latent_dim):
        super(Autoencoder2D, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            Conv2D(16, kernel_size=(3, 3), activation='relu'),
            Conv2D(8, kernel_size=(3, 3), activation='relu'),
            Conv2D(4, kernel_size=(3, 3), activation='relu'),
            Conv2D(2, kernel_size=(3, 3), activation='relu'),
            Flatten(),
            Dense(latent_dim, activation='relu'),
        ])
        self.decoder = Sequential([
            Dense(784, activation='relu'),
            Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder2D(latent_dim)


# In[ ]:


autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


# In[ ]:


autoencoder.fit(X_train3, X_train3,epochs=20, validation_data=(X_test3, X_test3))


# In[ ]:


encoded_train_imgs = autoencoder.encoder(X_train2).numpy()
encoded_test_imgs = autoencoder.encoder(X_test2).numpy()


# In[ ]:


X_train_df = pd.DataFrame(encoded_train_imgs, columns=['1', '2'])
X_train_df['class'] = y_train2
X_test_df = pd.DataFrame(encoded_test_imgs, columns=['1', '2'])
X_test_df['class'] = y_test2

X_2d_df = pd.concat([X_train_df, X_test_df])


# In[ ]:


plt.figure(figsize=(12, 6))
plt.scatter(X_2d_df['1'], X_2d_df['2'], c=X_2d_df['class'], s=50, cmap='viridis')
plt.title('Auto Encoder 2-D Representations')
plt.show()


# In[ ]:


X = X_2d_df[['1', '2']]
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)


# In[ ]:


plt.figure(figsize=(12, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('K-Means Predicitions')
plt.show()


# B3.3 Observation [1 marks]

# **Observations:**
# - It's really helpfull to reduce the dimensions of the data, results
# - PCA performed margianlly better than our autoencoder, but if we were to let it run for more epochs maybe we'll get the same if not better performance.

# In[ ]:




