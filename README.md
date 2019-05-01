
# Image classification with Convolutional Neural Networks

### By Ali Shannon

In this project, I am looking to make an image classifier that uses convolutional neural networks provided by Tensorflow using Keras. Convolutional networks look hierarchical patterns in image-type datasets which makes them superior when it comes to image classifications.

> ![CNN](https://cdn-images-1.medium.com/max/1600/1*NQQiyYqJJj4PSYAeWvxutg.png)
> [Credit](https://medium.freecodecamp.org/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050)

In this kind of neural network, each neuron only sees a 'patch' of the layer before it. This drastically improves 

This [dataset](https://www.kaggle.com/prasunroy/natural-images) contains images from 7 different categories.

- Airplane images obtained from http://host.robots.ox.ac.uk/pascal/VOC
- Car images obtained from https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- Cat images obtained from https://www.kaggle.com/c/dogs-vs-cats
- Dog images obtained from https://www.kaggle.com/c/dogs-vs-cats
- Flower images obtained from http://www.image-net.org
- Fruit images obtained from https://www.kaggle.com/moltean/fruits
- Motorbike images obtained from http://host.robots.ox.ac.uk/pascal/VOC
- Person images obtained from http://www.briancbecker.com/blog/research/pubfig83-lfw-dataset


### Preprocessing

Here I import the libraries I will use and get the dataset ready for tensorflow.


```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import glob, os

%matplotlib inline

np.random.seed(42) # uniform output across runs

tf.logging.set_verbosity(tf.logging.INFO) #turn off annoying error messages
```

Find the class names from the folder names


```python
classes = np.array(os.listdir('./'), dtype = 'O')
idx = np.argwhere([classes == '.ipynb_checkpoints', classes == 'CNN.ipynb'])
classes = np.delete(classes, idx)
```

Import images and filenames of images, then resize all images to 100x100 RGB 


```python
images = []; filenames = []

for cl in classes:
    for file in glob.glob('{}/*.jpg'.format(cl)): # import all jpg images from all classes 
        im = Image.open(file) 
        filenames.append(im.filename) # label images
        im = im.resize((100,100)) # resize all images
        images.append(im) # store images
```

Convert images to numpy arrays 


```python
X = np.array([np.array(image) for image in images])
```

Wherever the filename contains a certain class, assign that class to `l` and assign a given numerical value to `y` which will be our labels.


```python
labels = []

for file in filenames:
    for class_name in classes:
        if class_name in file:
            labels.append(class_name)
```


```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder().fit(labels)

y = encoder.transform(labels)
```

This checks the distributions of classes in the entire dataset


```python
unique_vals, counts = np.unique(labels, return_counts = True)

plt.figure(figsize = (7,4))
plt.bar(unique_vals, counts/counts.sum(), color = 'LightBlue');
vals = np.arange(0, 0.16, 0.03)
plt.yticks(vals, ["{:,.0%}".format(x) for x in vals])
plt.title('Distribution of images')
plt.show()
```


![png](README_files/README_13_0.png)


Let's look at the pictures of some images with their labels and encoding:


```python
plt.figure(figsize=(9,9))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    j = np.random.randint(len(images))
    plt.imshow(X[j], cmap=plt.cm.binary)
    plt.xlabel(labels[j] + ' : ' + str(y[j]))
plt.show()
```


![png](README_files/README_15_0.png)


### Convolutional Neural Network Training and Testing

First, we need to split the dataset into training and validation sets, the validation set is outside of the training set.


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
```

This is where all the magic happens. I have opted for 4 convolutional layer with their respective pooling layers. Added dropouts and Batch Normalization as they improved the accuracy. I have also added 256 and 64 neuron ReLU activated dense layers that learn from the convolutional flattened images.


```python
shapes = (100, 100, 3) # input shapes of all images

model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(5, 5), activation=tf.keras.activations.relu, input_shape=shapes),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis = 1),
    keras.layers.Dropout(rate = 0.25), 
    keras.layers.Conv2D(32, kernel_size=(5, 5), activation=tf.keras.activations.relu),
    keras.layers.AveragePooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis = 1),
    keras.layers.Dropout(rate = 0.25),
    keras.layers.Conv2D(32, kernel_size=(4, 4), activation=tf.keras.activations.relu),
    keras.layers.AveragePooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis = 1),
    keras.layers.Dropout(rate = 0.15),
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.keras.activations.relu),
    keras.layers.AveragePooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis = 1),
    keras.layers.Dropout(rate = 0.15),
    keras.layers.Flatten(),    
    keras.layers.Dense(256, activation=tf.keras.activations.relu,kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(64, activation=tf.keras.activations.relu,kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(len(classes), activation=tf.keras.activations.softmax)
])
model.compile(optimizer=keras.optimizers.Adam(),
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy'])

```

    WARNING:tensorflow:From D:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From D:\ProgramData\Anaconda3\lib\site-packages\tensorflow\python\keras\layers\core.py:143: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    


```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 15)
```

    Train on 4135 samples, validate on 2037 samples
    Epoch 1/15
    4135/4135 [==============================] - 14s 3ms/sample - loss: 1.4946 - acc: 0.5778 - val_loss: 1.2285 - val_acc: 0.6667
    Epoch 2/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.9499 - acc: 0.7674 - val_loss: 0.9231 - val_acc: 0.7806
    Epoch 3/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.8451 - acc: 0.8017 - val_loss: 0.7354 - val_acc: 0.82578385 - acc: 0
    Epoch 4/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.7340 - acc: 0.8225 - val_loss: 0.6443 - val_acc: 0.8596
    Epoch 5/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.6399 - acc: 0.8508 - val_loss: 0.7370 - val_acc: 0.8301
    Epoch 6/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.5805 - acc: 0.8665 - val_loss: 0.6340 - val_acc: 0.8483
    Epoch 7/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.5330 - acc: 0.8757 - val_loss: 0.5474 - val_acc: 0.8733
    Epoch 8/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.4874 - acc: 0.8958 - val_loss: 0.6072 - val_acc: 0.8576
    Epoch 9/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.4608 - acc: 0.8953 - val_loss: 0.4837 - val_acc: 0.8866
    Epoch 10/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.4162 - acc: 0.9103 - val_loss: 0.4662 - val_acc: 0.8866
    Epoch 11/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.4065 - acc: 0.9122 - val_loss: 0.4627 - val_acc: 0.8915
    Epoch 12/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.3519 - acc: 0.9224 - val_loss: 0.4428 - val_acc: 0.8856
    Epoch 13/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.3295 - acc: 0.9333 - val_loss: 0.4290 - val_acc: 0.8930
    Epoch 14/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.3252 - acc: 0.9328 - val_loss: 0.4773 - val_acc: 0.8802
    Epoch 15/15
    4135/4135 [==============================] - 7s 2ms/sample - loss: 0.3149 - acc: 0.9340 - val_loss: 0.3820 - val_acc: 0.9131
    




    <tensorflow.python.keras.callbacks.History at 0x29bf079b588>




```python
y_pred = model.predict(X_test)
```


```python
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    j = np.random.randint(len(X_test))
    plt.imshow(X_test[j], cmap=plt.cm.binary)
    plt.xlabel('Prediction: ' + classes[np.argmax(y_pred[j])])
plt.show()
```


![png](README_files/README_22_0.png)


Here is the confusion matrix for this model.


```python
from sklearn.metrics import confusion_matrix
import pandas as pd

conf = confusion_matrix(y_test, [np.argmax(y) for y in y_pred])

fig, ax = plt.subplots(figsize = (7,7))
ax.matshow(conf, cmap='Pastel2')

ax.set_ylabel('True Values')
ax.set_xlabel('Predicted Values', labelpad = 10)
ax.xaxis.set_label_position('top') 



ax.set_yticks(range(len(classes)))
ax.set_xticks(range(len(classes)))
ax.set_yticklabels(encoder.classes_)
ax.set_xticklabels(encoder.classes_)


for (i, j), z in np.ndenumerate(conf):
    ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

plt.show()
```


![png](README_files/README_24_0.png)


Here is the final accuracy of this model.


```python
val_loss, val_acc = model.evaluate(X_test, y_test)
```

    2037/2037 [==============================] - 1s 610us/sample - loss: 0.3820 - acc: 0.9131
    


```python
print('Final testing accuracy is ' + f'{val_acc *100} %')
```

    Final testing accuracy is 91.31075143814087 %
    

This has been more fun than the last 7 seasons of Game of Thrones.
