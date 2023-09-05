# TIPS:
## Batch-size, epochs and steps:
1. **Batch Size**: Due to the limited VRAM, start with a batch size of 2 or 4. Smaller batch sizes may be necessary if you encounter memory issues during training.
    
2. **Epochs**: Start with a smaller number of epochs, around 50 or 60. Monitor the training progress and validation metrics. You can decide to increase the number of epochs if the model performance is improving.
    
3. **Steps per Epoch**: The number of steps per epoch is typically the number of samples divided by the batch size. However, since you have a computational limitation, you can set it to a smaller value to reduce training time. For instance, if you have 1000 training samples and use a batch size of 4, you can set ``steps_per_epoch`` to 1000 // 4 = 250.
## Functions to load images and put them in dir.
## Extract string values from tensors
```
var = tensor.numpy().decode('utf8')
```
# 1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
## Introduction:
### Introduction to NN:
- Building a neural network of a single layer
```Python
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([keras.layers.dense(unite=1, input_shape=[1])])
# Dense: A layer of connected neurons
# Sequential: Stack of layers, in each layer we have one input tensor and one output tensor
model.compile(optimizer='sgd', loss='mse')
```
- Getting our data
```Python
import numpy as np
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
```
- Training the model
```Python
model.fit(xs,ys, epochs=500)
print(model.predict([10.0]))
```
>You will think that the result will $19$, but will me a value close to it.

*WHY? :* Because we have a small set of data to learn from, and also their is no guarantee that the relationship between `xs` and `ys` is linear, and also NN deals with probabilities more.

### Introduction to computer vision:
- Building a CNN
```Python
model = keras.Sequantial([
	keras.layers.Flatten(input_shape=(28,28)),
	# Flatten takes the input shape (28,28) and transform it into a simple linear array
	keras.layers.Dense(units=128, activation=tf.nn.relu),
	keras.layers.Dense(units=10, activation=tf.nn.softmax)
])
```
- Evaluation
```Python
model.evaluate(test_images, test_labels)
```
### Using Callbacks to control the training:
> Callback is used to stop training at certain point with a condition and save the model in that state
![[Pasted image 20230727144411.png]]
- Defining a Callback function
```Python
class myCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, log={}):
		if(log.get('loss') < 0.4):
			print('Loss is so low, cancelling the training')
			self.model.stop_training = True
```
![[Pasted image 20230727144747.png]]
## CNN:
### Convolution and Pooling
#### Convolution:
- The point from the convolution is making some characteristique of the image get emphasized
- It help us identify objects anywhere is the image
![[Pasted image 20230727152357.png]]
- Emphasizing horizontal lines in the images to make them pop-out
![[Pasted image 20230727152459.png]]
#### Pooling:
- It's a way to compress image size
![[Pasted image 20230727152820.png]]
#### Implementation:
- Building model
```Python
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (28,28,1)),
	# '64' represents the number of filters in the convolutional layers
	tf.keras.layers.MaxPooling2D(2, 2),
	# (2, 2) means that for every 4 pixels we take the max value
	tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
	tf.keras.layers.MaxPooling2D(2, 2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax'),
])
```
- Inspecting the model 
```Python
model.summary()
```
![[Pasted image 20230727153823.png]]
- Compiling binary classification
```Python
  model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])
```

### ImageDataGenerator
- We need to point ImageDataGenerator object to the directory and its automatically load labels for us according to the directory containing it.
![[Pasted image 20230728012607.png]]
```Python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(
	train_dir, # Point it at the dir containing the sub dirs
	target_size = (300, 300),
	batch_size = 128,
	class_mode = 'binary' # Means it will look for the difference between two classes
)
```
- Training with ImageDataGenerator
![[Pasted image 20230728160028.png]]
![[Pasted image 20230728161039.png]]
# 2. Convolutional Neural Networks in TensorFlow
## General:
- `model.layers` API will make me able to visualize the image created in each layer.
- A smaller size decreases the likelihood that the model will recognize all possible features during training.
## Augmentation techniques
- Help in increasing your dataset size and also increasing your model performance.
- Apply transformations on the images while its in memory and give it to the network to train on it, instead of applying transformations on the images and save those transformed images then loading them to the network
- https://keras.io/api/layers/preprocessing_layers/
- https://keras.io/api/data_loading/image/
```Python
# Updated image augmentation code
train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest'
)
```

- We need to have similar images to image augmentation for training in validation set

## Transfer Learning
- Download weights and add them
```Python
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = 'path/to/inceptionv3_weight_file.h5'

pre_trained_model = InceptionV3(
	input_shape = (150, 150, 3),
	include_top = False,
	weights = None
)
pre_trained_model.load_weights(local_weights_file)
```

- Lock the layers
```Python
for layer in pre_trained_model.layers:
	layer.trainable = False
```

- We can get the layer name from inspecting the summary of the model `pre_trained_model.summary()` 
```Python
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_later.output
```

- Create another output layers and merge them
```Python
from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(lr=0.0001),
				loss = 'binary_crossentropy',
				metrics = ['acc'])
```

## Exploring dropouts
- A good way to avoid overfitting
```Python
from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
# Adding dropout
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(lr=0.0001),
				loss = 'binary_crossentropy',
				metrics = ['acc'])
```

- Dropout is useful because in a deep network we can have a layer where the neighbor neurons can have similar weights
## Multiclass Classification
### Dataset
![[Pasted image 20230731135344.png]]
```Python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = train_datagen.flow_from_directory(
	train_dir, # Point it at the dir containing the sub dirs
	target_size = (300, 300),
	batch_size = 128,
	class_mode = 'categorical'
)
```
![[Pasted image 20230731135621.png]]
![[Pasted image 20230731135659.png]]
# 3. Natural Language Processing in TensorFlow
## Sentiment in text
### Tokenizer
- Split text into words (Encoding words in a sentence)
```Python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

# Define input sentences
sentences = [
    'i love my dog',
    'I, love my cat'
    ]
# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words = 100)
# Generate indices for each word in the corpus
tokenizer.fit_on_texts(sentences)
# Get the indices and print it
word_index = tokenizer.word_index
print(word_index)
```
> ![[Pasted image 20230731152530.png]]

- Tokenizer strip stopping words (. , ; : ! ? ..)
```Python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define your input texts
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
# Initialize the Tokenizer class
tokenizer = Tokenizer(num_words = 100)
# Tokenize the input sentences
tokenizer.fit_on_texts(sentences)
# Get the word index dictionary
word_index = tokenizer.word_index
# Generate list of token sequences
sequences = tokenizer.texts_to_sequences(sentences)
# Print the result
print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
```
- We need a lot of training data or we will end up in a situation like the one *my god my*
![[Pasted image 20230731155257.png]]
- To add a special sign to unseen words
```Python
tokenizer = Tokenizer(num_words = 100, oov_token='<OOV>')
```
> ![[Pasted image 20230731160227.png]]
> With more training data, we hope that we will have coverage for the unseen data.
### Padding
- You will usually need to pad the sequences into a uniform length because that is what your model expects. You can use the [pad_sequences](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences) for that. By default, it will pad according to the length of the longest sequence. You can override this with the `maxlen` argument to define a specific length. Feel free to play with the [other arguments](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences#args) shown in class and compare the result.
```Python
# Pad the sequences to a uniform length
padded = pad_sequences(sequences, maxlen=5)

# Print the result
print("\nPadded Sequences:")
print(padded)
```
> Example of result:
> ![[Pasted image 20230802110535.png]]

- If we wanted padding after the sentence:
![[Pasted image 20230802110610.png]]
- Truncation is the act of removing parts
![[Pasted image 20230802113310.png]]

## Word Embedding
### General
- Embedding is the concept of words and associated words being clustered in a vectorized dimensional space.
![[Pasted image 20230802153007.png]]
> ![[Pasted image 20230802153051.png]]

### Model
- The NN will try to predict the vector value of a word in a Embedded cluster
![[Pasted image 20230802161448.png]]
![[Pasted image 20230802161541.png]]
The second case will be faster than the flatten case; it a trade between accuracy and speed
![[Pasted image 20230802161709.png]]
