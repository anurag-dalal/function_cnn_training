# function_cnn_training
I have always facinatd by the idea that CNN can be used to approximate any function.  Here is a implementation of the same for visualization purpose a 2-variable function is used

## Install Dependecies
```unix
$pip install numpy
$pip install matplotlib
$pip install tensorflow
$pip install scikit-learn
```

## The Function to train
Here I have used a 2 variable function fro easy visualization and plotting.
```python
def fun(x,y):
    res1 = np.power(x, 2)+np.power(y, 2)<1
    res2 = np.power(x, 2)+np.power(y, 2)>0.5
    res3 = np.absolute(x-y)>0.15
    res = np.logical_and(res1, res2)
    res = np.logical_and(res, res3)
    return res.astype('uint8')
```
The function looks like:   
![Original Image](/images/original.png "Original Image")

Where the two classes are represented by two different colors.

## The Model
A simple Sequential model is build to train the function:  
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_14 (Dense)             (None, 64)                192       
_________________________________________________________________
dense_15 (Dense)             (None, 64)                4160      
_________________________________________________________________
flatten_14 (Flatten)         (None, 64)                0         
_________________________________________________________________
dropout_70 (Dropout)         (None, 64)                0         
_________________________________________________________________
dense_16 (Dense)             (None, 2)                 130       
=================================================================
Total params: 4,482
Trainable params: 4,482
Non-trainable params: 0
```

## Loss
After 15 epochs the losses are:  
```
Epoch 15/15
317/317 [==============================] - 1s 2ms/step - loss: 0.1399 - accuracy: 0.9550 - val_loss: 0.1043 - val_accuracy: 0.9747
```

## Accuracy
After 15 epochs the accuracy scores are:  
```
Test loss: 0.10299034416675568
Test accuracy: 0.9765999913215637
```

## Original distribution on test:

![Original Test Image](/images/originaltest.png "Original Test Image")
## Predicted distriution on test:

![Predicted Image](/images/predicted.png "predicted Image")
