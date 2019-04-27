from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.optmizers import SGD
from keras.utils import np_utils
from sklearn import datasets

# Config values
num_classes = 9
img_depth = 1
img_height = 28
img_width = 28
# Creating the LeNet model
model = Sequential()
# Adding the first convolutional layer
model.add(Convolution2D(20, 5, 5, border_mode="same",
                        input_shape=(img_depth, img_height, img_width)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Adding the second convolutional layer
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Adding the fully connected layers
model.add(Flatten())
model.add(Dense(500))
# Load MNIST data
model.add(Activation("relu"))
# Adding a softmax layer
model.add(Dense(num_classes))
model.add(Activation("softmax"))
mnist = datasets.fetch_mldata("MNIST Original")
# MNIST data is a flat array of size 784.
# We need to reshape it be in 28x28 images as we have to feed it to a convolutional layer
mnist.data = mnist.data.reshape((mnist.data.shape[0], 28, 28))

mnist.data = mnist.data[:, np.newaxis, :, :]
mnist.data = mnist.data / 255.0  # Normalize the images to [0, 1.0]
# Split the data into train and test set
train_data, test_data, train_label, test_label =
train_test_split(minist.data, mnist.target, test_size=0.25)
train_label = np_utils.to_categorical(train_label, 10)
test_label = np_utils.to_categorical(test_label, 10)
# Set the loss funtions and evaluation metrics
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.0001),
              metrics=["accuracy"])
# Train the LeNet model
model.fit(train_data, train_label, batch_size=32, no_epoch=30, verbose=1)
# Test the model
loss, accuracy = model.evaluate(test_data, test_label, batch_size=64,
                                verbose=1)
print("Accuracy: %".format(accuracy * 100))
