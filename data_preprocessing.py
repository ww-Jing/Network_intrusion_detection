import numpy as np # linear algebra
import pandas as pd 
import keras
from keras.preprocessing import image
from sklearn.preprocessing import Normalizer, MinMaxScaler

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



def Data_Preprocessing(test_size_num, df):
    '''
    df.replace(np.inf,np.NaN,inplace=True)
    df.dropna(inplace=True)

    df = df.drop(columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 
                     'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg'])#, 'Bwd Blk Rate'])
    
    df = df.replace({'DDOS attack-HOIC':'DDOS','DDOS attack-LOIC-UDP':'DDoS'})
    df = df.replace({'DoS attacks-GoldenEye':'DoS','DoS attacks-Hulk':'DoS',
                    'DoS attacks-SlowHTTPTest':'DoS','DoS attacks-Slowloris':'DoS'})
    df = df.replace({'FTP-BruteForce':'BruteForce','SSH-Bruteforce':'BruteForce'})
    '''
    
    
    #rows_to_drop = df[df['Label'] == 'Label'].index
    #df = df.drop(rows_to_drop, inplace=True)
    df = df[df['Label'] != 'Label']

    df = df.drop(columns=['Timestamp', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port'])
    df = df.drop(columns=['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'])

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    #df['Flow Byts/s'].fillna(df['Flow Byts/s'].mode()[0], inplace=True)
    #df['Flow Pkts/s'].fillna(df['Flow Pkts/s'].mode()[0], inplace=True)

    df = df.drop_duplicates()

    df = df.replace({'FTP-BruteForce':'Bruteforce','SSH-Bruteforce':'Bruteforce'})
    df = df.replace({'DoS attacks-GoldenEye':'DoS','DoS attacks-Slowloris':'DoS', 'DoS attacks-SlowHTTPTest':'DoS', 'DoS attacks-Hulk':'DoS'})
    df = df.replace({'Brute Force -Web':'Web_Attack', 'Brute Force -XSS':'Web_Attack', 'SQL Injection':'Web_Attack'})
    df = df.replace({'Infilteration':'Infiltration'})
    df = df.replace({'Bot':'Botnet'})
    df = df.replace({'DDoS attacks-LOIC-HTTP':'DDoS', 'DDOS attack-LOIC-UDP':'DDoS','DDOS attack-HOIC':'DDoS'})
    attacks = ['Benign', 'BruteForce', 'DoS', 'Web Attack', 'Infilteration', 'Bot', 'DDoS']
    
    
    input_len = df.shape[1]-1
    attacks = ['Benign', 'BruteForce', 'DoS', 'Infilteration', 'Bot', 'DDoS']
    
    class_num = len(attacks)
    for count, attack in enumerate(attacks):
        df.replace(to_replace=attack, value=count, inplace=True)
    
    X = df.iloc[:,1:input_len].values 
    y = df.iloc[:,input_len].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=420, test_size=test_size_num)

    scaler_1, scaler_2 = Normalizer().fit(X_train), Normalizer().fit(X_test)
    X_train, X_test = scaler_1.transform(X_train), scaler_2.transform(X_test)
    y_train, y_test = to_categorical(np.array(y_train)), to_categorical(np.array(y_test))
    
    # reshape input to be [samples, time steps, features]
    X_train_cnn = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_cnn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, X_train_cnn, X_test_cnn, y_train, y_test, class_num, input_len