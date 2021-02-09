'''
I have always facinatd by the idea
that CNN can be used to approximate
any function.

Here is a implementation of the same
for visualization purpose a 2-variable function is used 
'''
# importing the necessary libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# defining the function, the function is a 
# ring with certain portion of the ring also belonging to
# another class
def fun(x,y):
    res1 = np.power(x, 2)+np.power(y, 2)<1
    res2 = np.power(x, 2)+np.power(y, 2)>0.5
    res3 = np.absolute(x-y)>0.15
    res = np.logical_and(res1, res2)
    res = np.logical_and(res, res3)
    return res.astype('uint8')

# creating sample points of x and y
x = np.random.rand(50000)
y = np.random.rand(50000)
res = fun(x,y)

# concatinnating the points in a single array
# since our data points are in the range [0 1]
# we dont need to normalize
data = [x, y]
data = np.array(data, dtype = 'float32')
data = data.T
# splitting in train and test set
X_train, X_test, y_train, y_testt =train_test_split(data, res, test_size=0.1, random_state=42)
plt.figure(figsize=(3,3))
plt.scatter(x,y, c=res)
plt.title('Original distribution')
plt.show()
# Model / data parameters
num_classes = 2
input_shape = (None, 2)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_testt, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=[2]),
        layers.Dense(64, activation="swish"),
        layers.Dense(64, activation="relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
batch_size = 128
epochs = 15
# compiling and fitting the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
# evaluating the model
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# getting the prediction
y_pred = model.predict(X_test)
yp = []
# converting the prediction probability into classes
for row in y_pred:
    if(row[0]>row[1]):
        yp.append(0)
    else:
        yp.append(1)
      
# plotting the predicted and original test points
yp = np.array(yp, dtype = 'uint8')
plt.figure(figsize=(3,3))
plt.scatter(X_test[:,0],X_test[:,1], c=yp)
plt.title('Predicted distribution on test')
plt.show()
plt.figure(figsize=(3,3))
plt.scatter(X_test[:,0],X_test[:,1], c=y_testt)
plt.title('Original distribution on test')
plt.show()