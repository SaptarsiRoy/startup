#Load the packages
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.datasets import mnist
from keras.utils import np_utils
import keras

#Loading the mnist dataset
(x_train, y_train), (x_test, y_test)  = mnist.load_data()

#Storing the number of rows and columns
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

#Shaping our data as we need the shape of (60000,28,28,1) for model training 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

#Storing the shape of input to be fed
input_shape = (img_rows, img_cols, 1)

# change our image type to float32 data type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize our data by changing the range from (0 to 255) to (0 to 1)
x_train /= 255
x_test /= 255

# Now we one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

#Creating LeNet model for mnist dataset
# create model
model = Sequential()

# 2 sets of CRP (Convolution, RELU, Pooling)
model.add(Conv2D(20, (5, 5), padding = "same", input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(50, (5, 5), padding="same"))
model.add(Conv2D(50, (5, 5), padding = "same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))

# Fully connected layers (w/ RELU)
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# Softmax (for classification)
model.add(Dense(num_classes))
model.add(Activation("softmax"))
model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

summary = model.summary()
print("""<pre>{0}</pre>""".format(summary))
#Now we have to train our model
# Training Parameters
batch_size = 128
epochs = 10

print("Trainig data data")

history = model.fit(x_train, y_train, verbose = 0 ,batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test), shuffle = True)

#After trainig model we have to save it for downloading
#Saving model 
model.save("mnist_LeNet.h5")

#Now testing the accuracy of the model
scores = model.evaluate(x_test, y_test, verbose=0)

#Printing the scores
print("Test Loss: ", scores[0])
print("Test accuracy: ", scores[1])
