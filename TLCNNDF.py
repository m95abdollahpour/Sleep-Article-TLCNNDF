from sklearn.model_selection import train_test_split
from keras import optimizers
from keras.models import  Model
from keras.layers import Conv2D, concatenate
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense ,Dropout,  Input
from sklearn.metrics import classification_report, confusion_matrix




# the implementation of transfer learning convolutional neural network for data fusion
#TLCNNDF


x_train, x_test, y_train, y_test = train_test_split(data1, hypn, test_size=0.2,shuffle=True,
                                                        random_state=0)



input1 = Input(shape = (64, 64, 3))
conv1 = Conv2D(32, (2, 2) , activation = 'relu', name='conv1eog')(input1) 
max1 = MaxPooling2D(pool_size = (3, 3))(conv1) 
drop1 = Dropout(0.5)(max1)
conv2 = Conv2D(64, (2, 2), activation = 'relu', name='conv2eog')(drop1) 
max2 = MaxPooling2D(pool_size = (2,2))(conv2) 
conv3 = Conv2D(128, (2, 2), activation = 'relu', name='conv3eog')(max2) 
drop3 = Dropout(0.5)(conv3)
flat = Flatten()(drop3)
dense1 = Dense(units = 128, activation = 'relu')(flat) 
dropd = Dropout(0.4)(dense1)
dense2 = Dense(units =5, activation = 'softmax')(dropd)
subnetwork1 = Model(inputs = input1, outputs = dense2)
subnetwork1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
historyy=subnetwork1.fit(x_train[:,:,:,3:6], y_train,
                        validation_data=(x_test[:,:,:,3:6],y_test), batch_size=32, epochs=30)

subnetwork1.save_weights('subnetwork1.h5')


input1 = Input(shape = (64, 64, 3))
conv1 = Conv2D(64, (2, 2) , activation = 'relu', name='conv1eeg')(input1) 
max1 = MaxPooling2D(pool_size = (3, 3))(conv1) 
drop1 = Dropout(0.5)(max1)
conv2 = Conv2D(128, (2, 2), activation = 'relu', name='conv2eeg')(drop1) 
max2 = MaxPooling2D(pool_size = (2,2))(conv2) 
drop2 = Dropout(0.5)(max2)
flat = Flatten()(drop2)
dense1 = Dense(units = 128, activation = 'relu')(flat)
dropd = Dropout(0.4)(dense1)
dense2 = Dense(units =5, activation = 'softmax')(dropd)
subnetwork2 = Model(inputs = input1, outputs = dense2)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
subnetwork2.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
hhistory=subnetwork2.fit(x_train[:,:,:,0:3], y_train,
                        validation_data=(x_test[:,:,:,0:3],y_test), batch_size=32, epochs=30)

subnetwork2.save_weights('subnetwork2.h5')




input1 = Input(shape = (64, 64, 3))
conv1 = Conv2D(32, (2, 2) , activation = 'relu', name='conv1eog',trainable=False)(input1) 
max1 = MaxPooling2D(pool_size = (3, 3))(conv1)
drop1 = Dropout(0.5)(max1)
conv2 = Conv2D(64, (2, 2), activation = 'relu', name='conv2eog',trainable=False)(drop1) 
max2 = MaxPooling2D(pool_size = (2,2))(conv2) 
conv3 = Conv2D(128, (2, 2), activation = 'relu', name='conv3eog',trainable=False)(max2) 
drop3 = Dropout(0.5)(conv3)
flat1 = Flatten()(drop3)
input2 = Input(shape = (64, 64, 3))   
conv11 = Conv2D(64, (2, 2) , activation = 'relu', name='conv1eeg',trainable=False)(input2) 
max11 = MaxPooling2D(pool_size = (3, 3))(conv11) 
drop11 = Dropout(0.5)(max11)
conv22 = Conv2D(128, (2, 2), activation = 'relu', name='conv2eeg',trainable=False)(drop11) 
max22 = MaxPooling2D(pool_size = (2,2))(conv22) 
drop22 = Dropout(0.5)(max22)
flat22 = Flatten()(drop22)
fused1 = concatenate([flat1, flat22])
dense1 = Dense(units = 100,activation = 'relu')(fused1) 
dropd = Dropout(0.7)(dense1)
dense2 = Dense(units =5, activation = 'softmax')(dropd)
fusion_model = Model(inputs = [input1, input2], outputs = dense2)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
fusion_model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
fusion_model.load_weights('subnetwork2.h5', by_name=True)
fusion_model.load_weights('subnetwork1.h5', by_name=True)
history143=fusion_model.fit([x_train[:,:,:,3:6],x_train[:,:,:,0:3]], y_train,
                        validation_data=([x_test[:,:,:,3:6],x_test[:,:,:,0:3]],y_test), batch_size=64, epochs=50)

fusion_model.save("main_model.h5")



