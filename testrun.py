import tflearn
import tflearn.datasets.mnist as mnist
from tflearn.layers.core import fully_connected,input_data
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.estimator import regression
import time

trainX,trainY,testX,testY = mnist.load_data(one_hot=True)
trainX=trainX.reshape([-1,28,28,1])
testX=testX.reshape([-1,28,28,1])
model = input_data(shape=[None,28,28,1])
model = conv_2d(model,30,3,activation='relu')
model = max_pool_2d(model,4)
model = conv_2d(model,30,3,activation='relu')
model = max_pool_2d(model,4)
model = fully_connected(model,500,activation='relu')
model = fully_connected(model,500,activation='relu')
model = fully_connected(model,10,activation='softmax')
model = regression(model)
model = tflearn.DNN(model,tensorboard_verbose=3)
start = time.time()
model.fit(X_inputs=trainX,Y_targets=trainY,n_epoch=5,
          validation_set=(testX,testY),show_metric=True)
print("Time taken:",time.time()-start,'seconds')