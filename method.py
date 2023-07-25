
#def __init__(self):
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Embedding, SpatialDropout1D, BatchNormalization, MaxPooling1D
from keras.layers import SimpleRNN, LSTM, Conv1D, Reshape
from keras.utils import to_categorical



def dnn(X_train, y_train, X_test, y_test, class_num, input_len, drop_num):
    input_len = input_len - 1 
    model = Sequential()
    model.add(Dense(64, input_shape=(1, input_len)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))
    model.add(Dense(64, input_shape=(1, input_len)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))
    model.add(Dense(64, input_shape=(1, input_len)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))
    model.add(Dense(32, input_shape=(1, input_len)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))
    model.add(Dense(32, input_shape=(1, input_len)))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))
    model.add(Flatten())
    #model.add(Dense(class_num,activation='sigmoid'))
    model.add(Dense(class_num,activation='softmax'))

    #batch_input_shape=(None, 72, 1)
    #model.build(batch_input_shape)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model


def simplernn(X_train, y_train, X_test, y_test, class_num, input_len, drop_num):
    model = Sequential()
    
    model.add(SimpleRNN(64,input_shape=(1, 72), return_sequences= True ))  
    model.add(BatchNormalization(synchronized=True))
    model.add(Dropout(drop_num))
    model.add(SimpleRNN(64,return_sequences=True)) 
    model.add(BatchNormalization(synchronized=True))
    model.add(Dropout(drop_num))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(BatchNormalization(synchronized=True)) 
    model.add(Dropout(drop_num))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(BatchNormalization(synchronized=True)) 
    model.add(Dropout(drop_num))
    model.add(SimpleRNN(32, return_sequences=False))
    model.add(BatchNormalization(synchronized=True)) 
    model.add(Dropout(drop_num))
    #model.add(Dense(class_num,activation='sigmoid'))
    model.add(Dense(class_num,activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    return model


def cnn(X_train, y_train, X_test, y_test, class_num, input_len, drop_num):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))
    model.add(Flatten())
    #model.add(Dense(class_num,activation='sigmoid'))
    model.add(Dense(class_num,activation='softmax'))

    batch_input_shape=(None, 72, 1)
    model.build(batch_input_shape)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    return model

def lstm(X_train, y_train, X_test, y_test, class_num, input_len, drop_num):
    
    model = Sequential()

    model.add(LSTM(64,input_shape=(1, 72), return_sequences= True ))  
    model.add(BatchNormalization(synchronized=True))
    model.add(Dropout(drop_num))
    model.add(LSTM(64, return_sequences= True ))  
    model.add(BatchNormalization(synchronized=True))
    model.add(Dropout(drop_num))
    model.add(LSTM(64, return_sequences= True ))  
    model.add(BatchNormalization(synchronized=True))
    model.add(Dropout(drop_num))
    model.add(LSTM(32, return_sequences= True ))  
    model.add(BatchNormalization(synchronized=True))
    model.add(Dropout(drop_num))
    model.add(LSTM(32, return_sequences= True ))  
    model.add(BatchNormalization(synchronized=True))
    model.add(Dropout(drop_num))
    model.add(Flatten())
    #model.add(Dense(class_num,activation='sigmoid'))
    model.add(Dense(class_num,activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return model


def cnnrnn(X_train, y_train, X_test, y_test, class_num, input_len, drop_num):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))

    model.add(Reshape(target_shape=(1, 72)))

    model.add(SimpleRNN(64,input_shape=(1, 72), return_sequences= True ))  
    model.add(BatchNormalization(synchronized=True))
    model.add(Dropout(drop_num))
    model.add(SimpleRNN(64,return_sequences=True)) 
    model.add(BatchNormalization(synchronized=True))
    model.add(Dropout(drop_num))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(BatchNormalization(synchronized=True)) 
    model.add(Dropout(drop_num))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(BatchNormalization(synchronized=True)) 
    model.add(Dropout(drop_num))
    model.add(SimpleRNN(32, return_sequences=False))
    model.add(BatchNormalization(synchronized=True)) 
    model.add(Dropout(drop_num))
    model.add(Flatten())
    #model.add(Dense(class_num,activation='sigmoid'))
    model.add(Dense(class_num,activation='softmax'))

    #batch_input_shape=(None, 72, 1)
    #model.build(batch_input_shape)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    return model


def cnnlstm(X_train, y_train, X_test, y_test, class_num, input_len, drop_num):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))

    model.add(Reshape(target_shape=(1, 72)))

    model.add(LSTM(64,input_shape=(1, 72), return_sequences= True ))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))
    model.add(LSTM(64, return_sequences= True ))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))
    model.add(LSTM(64, return_sequences= True ))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))

    model.add(LSTM(32, return_sequences= True ))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))

    model.add(LSTM(32, return_sequences= True ))
    model.add(BatchNormalization())
    model.add(Dropout(drop_num))

    #model.add(Dense(class_num,activation='sigmoid'))
    model.add(Dense(class_num,activation='softmax'))

    #batch_input_shape=(None, 72, 1)
    #model.build(batch_input_shape)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    return model