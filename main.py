import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import keras
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from keras import callbacks
from keras.callbacks import CSVLogger
from sklearn.metrics import classification_report

from method import dnn, cnn, simplernn, lstm
from data_preprocessing import Data_Preprocessing
from parameter import Para

para = Para()
para.batch_size_num = 50
para.epochs_num = 10 
para.test_size_num  = 0.2
para.drop_num = 0.1

#df = pd.read_csv('cic_ids_2018.csv') 

folder_path = 'Data'
all_data = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):  
        file_path = os.path.join(folder_path, filename)
        dfe = pd.read_csv(file_path)  
        all_data.append(dfe) 
df = pd.concat(all_data, ignore_index=True)

X_train, X_test, X_train_cnn, X_test_cnn, y_train, y_test, class_num, input_len = Data_Preprocessing(para.test_size_num, df)

model  = dnn(X_train, y_train, X_test, y_test, class_num, input_len, para.drop_num)
#model = simplernn(X_train, y_train, X_test, y_test, class_num, input_len, para.drop_num)
#model = cnn(X_train_cnn, y_train, X_test_cnn, y_test, class_num, input_len, para.drop_num)
#model = lstm(X_train, y_train, X_test, y_test, class_num, input_len, para.drop_num)
#model  = cnnrnn(X_train, y_train, X_test, y_test, class_num, input_len, para.drop_num)
#model  = cnnlstm(X_train, y_train, X_test, y_test, class_num, input_len, para.drop_num)

csv_logger = CSVLogger('multiclass_Genlstm40.csv',separator=',', append=False)
history= model.fit(X_train, y_train, batch_size=para.batch_size_num, epochs = para.epochs_num, validation_data=(X_test, y_test),callbacks=[csv_logger])
loss, accuracy = model.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
#y_pred = model.predict_classes(X_test)
predict_x = model.predict(X_test) 
y_pred = np.argmax(predict_x,axis=1)
y_test=np.argmax(y_test, axis=1)
print(y_test.shape, y_pred.shape)
print(y_test[:5], y_pred[:5])
print(classification_report(y_test, y_pred))