import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from io import StringIO
import os

import pystan
from fbprophet import Prophet

from numpy import array
from numpy import hstack
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
from hyperopt.pyll.base import scope
import random


def supervi_format_multi_imputs_fr(donnes_train, n_steps, n_steps_out):
    X_train, y_predict = list(), list()
    for i in range(len(donnes_train)):
        # on cherche l'index de fin
        end_ix = i + n_steps
        out_end_ix = end_ix + n_steps_out-1

        # condition d'arrêt de la boucle : sinon valeurs NA 
        if out_end_ix + 1  > len(donnes_train):
            break
        seq_x, seq_y = donnes_train[i:end_ix,:], donnes_train[end_ix:out_end_ix + 1,0]
        X_train.append(seq_x)
        y_predict.append(seq_y)
    return array(X_train), array(y_predict)

def build_space(len_train, n_step_out,val_target,liste_feature,data):
    #nb_features_rand = random.randint(0, len(liste_feature))
    space = {'choice': hp.choice('num_layers',
                          [ {'layers':'one'},
                            {'layers':'two',
                            'units2' :  scope.int(hp.quniform('l2_units2',5,100,5))}
                            #{'layers':'three',
                            #'units2': scope.int(hp.quniform('units2',5,30,5)), 
                            #'units3' : scope.int(hp.quniform('units3',5,20,5)),
                            #'dropout2' : hp.uniform('dropout2', .25,.75),
                            #'dropout3' : hp.uniform('dropout3', .25,.75)}
                            ]),
                'units1': scope.int(hp.quniform('units',5,50,5)),
                'dropout1': hp.uniform('dropout1', .05,.2),
                'batch_size' : hp.uniform('batch_size', 28,91),
                #'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
                'optimizer': hp.choice('optimizer',['adam']),
                'n_steps_out' : n_step_out,
                'train' : len_train,
                'look' : scope.int(hp.quniform('look', 5, 25, 5)),
                'epoch' : 2000,
                'target' : val_target,
                'batch_size' : scope.int(hp.quniform('batch_size',5,91,15)),
                'regularisation' : hp.choice('regularisation',[None, 'l1', 'l2']),
                'nb_features_random' : hp.randint('nb_features_random',
                                                  len(liste_feature) + 1),
                'liste_feature' : liste_feature,
                'data' : data
                }
    return space

def build_space_features(len_train, n_step_out,val_target,liste_feature,data):
    #nb_features_rand = random.randint(0, len(liste_feature))
    space = {'choice': hp.choice('num_layers',
                          [ {'layers':'one'},
                            {'layers':'one',
                            'units2' :  scope.int(hp.quniform('l2_units2',5,100,5))}
                            #{'layers':'three',
                            #'units2': scope.int(hp.quniform('units2',5,30,5)), 
                            #'units3' : scope.int(hp.quniform('units3',5,20,5)),
                            #'dropout2' : hp.uniform('dropout2', .25,.75),
                            #'dropout3' : hp.uniform('dropout3', .25,.75)}
                            ]),
                'units1': 25,
                'dropout1': 0.15,
                #'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
                'optimizer': hp.choice('optimizer',['adam']),
                'n_steps_out' : n_step_out,
                'train' : len_train,
                'look' : 5,
                'epoch' : 2000,
                'target' : val_target,
                'batch_size' : 15,
                'regularisation' : 'l2',
                'nb_features_random' : hp.randint('nb_features_random',
                                                  len(liste_feature) + 1),
                'liste_feature' : liste_feature,
                'data' : data
                }
    return space

def deeplearning_mlp(space):
   
    n_steps_out = space['n_steps_out']
    train = space['train']
    look = space['look']
    epoch = space['epoch']
    batch_size = space['batch_size']
    regularisation = space['regularisation']
    data_fr_lisse = space['target']
    liste_features = space['liste_feature']
    optimizer = space['optimizer']
    nb_neurones_input = space['units1'] 
    nb_neurones_interm = [0 ,0]
    data = space['data']
    
    if space['choice']['layers']== 'two':
        nb_neurones_interm = [space['choice']['units2'], 0]
    
    if  space['choice']['layers']== 'three':
        nb_neurones_interm = [space['choice']['units2'], space['choice']['units3']]
    
    # On ajoute l'indicateur saisonier pour le covid
    x = np.arange(len(data_fr_lisse))
    feature_saison = np.sin(x*np.pi/12.)
    feature_saison = feature_saison.reshape(len(data_fr_lisse),1)
    
    
    
    # Scaling des inputs:
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target = scaler_target.fit(data_fr_lisse.reshape(-1, 1))
    normalized_target = scaler_target.transform(data_fr_lisse.reshape(-1, 1))
    data_fr_lisse = normalized_target.reshape(len(data_fr_lisse),1)
    
    Data_all_lisse = hstack((data_fr_lisse ,feature_saison))
    
    # On ajoute la liste des features une par une avec un scaling
    for i in range(len(liste_features)):
        # extraire la feauture 
        feature_ = data[liste_features[i]].values
        # scaler la feature
        scaler_ = MinMaxScaler(feature_range=(0, 1))
        scaler_ = scaler_.fit(feature_.reshape(-1, 1))
        feature_ = scaler_.transform(feature_.reshape(-1, 1)) 
        feature_ = normalized_target.reshape(len(data_fr_lisse),1)
        # hstack avec les autres features
        Data_all_lisse = hstack((Data_all_lisse ,feature_))
    
    Data_train = Data_all_lisse[:train,:]
    
    #Data_predict = Data_all_lisse[train,:]
    len_Data_train = len(Data_train)
    n_steps = look
    #n_steps_out = 29
    Data_train_SVF, Data_predict_SVF = supervi_format_multi_imputs_fr(Data_train,
                                                                      n_steps,
                                                                      n_steps_out)
  
    n_input = Data_train_SVF.shape[1] * Data_train_SVF.shape[2]
    #print(n_input)
    Data_train_SVF = Data_train_SVF.reshape((Data_train_SVF.shape[0], n_input))
    #print(Data_train_SVF)
    model = Sequential()
    model.add(Dense(nb_neurones_input, activation='relu', input_dim=n_input,
                    kernel_regularizer=regularisation))
    model.add(Dropout(rate = space['dropout1']))
    for i in range(len(nb_neurones_interm)):
        if nb_neurones_interm[i] != 0 :
            model.add(Dense(nb_neurones_interm[i], activation='relu',
                            input_dim=n_input,kernel_regularizer=regularisation))
            model.add(Dropout(space['choice']['dropout2']))
            
    #model.add(Dense(4))
    model.add(Dense(n_steps_out,kernel_regularizer=regularisation))
    model.compile(optimizer='adam', loss='mse')
   
    for i in range(len(Data_predict_SVF)):
        Data_predict_SVF[i] = Data_predict_SVF[i].tolist()
    
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)
    result = model.fit(Data_train_SVF, Data_predict_SVF, epochs=epoch,
              verbose=0, batch_size = batch_size,
              validation_split=0.1,
              callbacks=[es,TqdmCallback(verbose=1)],
              shuffle = True)
    
    validation_loss = np.amin(result.history['val_loss']) 
    print('Best validation loss of epoch:', validation_loss)
    
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model, 'params': space}


def deeplearning_lstm(space):
    n_steps_out = space['n_steps_out']
    train = space['train']
    look = space['look']
    epoch = space['epoch']
    batch_size = space['batch_size']
    regularisation = space['regularisation']
    data_fr_lisse = space['target']
    liste_features = space['liste_feature']
    optimizer = space['optimizer']
    nb_neurones_input = space['units1'] 
    nb_neurones_interm = [0 ,0]
    data = space['data']
    
    if space['choice']['layers']== 'two':
        nb_neurones_interm = [space['choice']['units2'], 0]
    
    if  space['choice']['layers']== 'three':
        nb_neurones_interm = [space['choice']['units2'], space['choice']['units3']]
        
    # On ajoute l'indicateur saisonier pour le covid
    x = np.arange(len(data_fr_lisse))
    feature_saison = np.sin(x*np.pi/12.)
    feature_saison = feature_saison.reshape(len(data_fr_lisse),1)
    
    # Scaling des inputs:
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target = scaler_target.fit(data_fr_lisse.reshape(-1, 1))
    normalized_target = scaler_target.transform(data_fr_lisse.reshape(-1, 1))
    data_fr_lisse = normalized_target.reshape(len(data_fr_lisse),1)
    
    Data_all_lisse = hstack((data_fr_lisse ,feature_saison))
    
    # On ajoute la liste des features une par une avec un scaling
    for i in range(len(liste_features)):
        # extraire la feauture 
        feature_ = data[liste_features[i]].values
        # scaler la feature
        scaler_ = MinMaxScaler(feature_range=(0, 1))
        scaler_ = scaler_.fit(feature_.reshape(-1, 1))
        feature_ = scaler_.transform(feature_.reshape(-1, 1))
        feature_ = normalized_target.reshape(len(data_fr_lisse),1)
        # hstack avec les autres features
        Data_all_lisse = hstack((Data_all_lisse ,feature_))
    
    
    Data_train = Data_all_lisse[:train,:]
    #Data_predict = Data_all_lisse[train,:]
    len_Data_train = len(Data_train)
    n_steps = look
    #n_steps_out = 29
    Data_train_SVF, Data_predict_SVF = supervi_format_multi_imputs_fr(Data_train,
                                                                      n_steps,
                                                                      n_steps_out)
  
    n_features = Data_train_SVF.shape[2]
    
    model = Sequential()
    if nb_neurones_interm[0] == 0:
        return_sequences=False
    else:
        return_sequences=True
        
    model.add(LSTM(nb_neurones_input, activation='relu',
                   return_sequences=return_sequences,
                   input_shape=(look, n_features)))
    
    if nb_neurones_interm[0] != 0:
        model.add(LSTM(nb_neurones_interm[0], activation='relu'))
        
    if nb_neurones_interm[1] != 0:
        model.add(Dense(nb_neurones_interm[1], activation='relu',
                        kernel_regularizer=regularisation))
    
    model.add(Dense(n_steps_out,kernel_regularizer=regularisation))

    model.compile(optimizer='adam', loss='mse') 
   
    for i in range(len(Data_predict_SVF)):
        Data_predict_SVF[i] = Data_predict_SVF[i].tolist()
    #st.write('avant fit')
    
    
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)
    result = model.fit(Data_train_SVF, Data_predict_SVF, epochs=epoch,
              verbose=0, batch_size = batch_size,
              validation_split=0.1,
              callbacks=[es,TqdmCallback(verbose=1)],
              shuffle = True)
    
    validation_loss = np.amin(result.history['val_loss']) 
    print('Best validation loss of epoch:', validation_loss)
    
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model, 'params': space}

    

def deeplearning_mlp_hp(params):
    
    target = params['target']
    feature_1 = params['feature_1']
    feature_2 = params['feature_2']
    
    scaler_target = MinMaxScaler(feature_range=(-5, 5))
    scaler_covid = MinMaxScaler(feature_range=(0, 1))
    scaler_trend = MinMaxScaler(feature_range=(0, 1))
    
    scaler_target = scaler_target.fit(target.reshape(-1, 1))
    scaler_covid = scaler_covid.fit(feature_1.reshape(-1, 1))
    scaler_trend = scaler_trend.fit(feature_2.reshape(-1, 1))
    
    normalized_target = scaler_target.transform(target.reshape(-1, 1))
    normalized_covid = scaler_covid.transform(feature_1.reshape(-1, 1))
    normalized_trend = scaler_trend.transform(feature_2.reshape(-1, 1))
    
    # On ajoute l'indicateur saisonier pour le covid
    x = np.arange(len(target))
    feature_saison = np.sin(x*np.pi/12.)
    feature_saison = feature_saison.reshape(len(target),1)
    
    target = normalized_target.reshape(len(target),1)
    feature_1 = normalized_covid.reshape(len(target),1)
    feature_2 = normalized_trend.reshape(len(target),1)
    

    Data_all_lisse = hstack((target, feature_1, feature_2,feature_saison))
    Data_train = Data_all_lisse[:params['train'],:]
    
    #Data_predict = Data_all_lisse[train,:]
    len_Data_train = len(Data_train)
    n_steps = params['look']
    #params['n_steps_out'] = 29
    Data_train_SVF, Data_predict_SVF = supervi_format_multi_imputs_fr(Data_train,
                                                                      n_steps,
                                                                      params['n_steps_out'])
  
    n_input = Data_train_SVF.shape[1] * Data_train_SVF.shape[2]
    Data_train_SVF = Data_train_SVF.reshape((Data_train_SVF.shape[0], n_input))
    model = Sequential()
    model.add(Dense(params['nb_neurones_input'], activation='relu', input_dim=n_input,
                    kernel_regularizer=params['regularisation']))
    model.add(Dense(params['n_steps_out'],kernel_regularizer=params['regularisation']))
    model.compile(optimizer='adam', loss='mse')
    
    for i in range(len(Data_predict_SVF)):
        Data_predict_SVF[i] = Data_predict_SVF[i].tolist()

    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=40)
    result = model.fit(Data_train_SVF, Data_predict_SVF, epochs=params['epoch'],
                       verbose=0, batch_size = params['batch_size'],
                       validation_split=0.1,
                       callbacks=[es,TqdmCallback(verbose=1)])
    validation_loss = np.amin(result.history['val_loss']) 
    print('Best validation loss of epoch:', validation_loss)
    return {'loss': validation_loss, 'status': STATUS_OK, 'model': model, 'params': params}

#%% les fonctions avec objectif

def eval_error_volume(y, y_pred):
    sum_y = np.sum(y)
    sum_y_pred = np.sum(y_pred)
    return abs((sum_y-sum_y_pred)/sum_y)*100

def objectif_volume(model,
                    look_back,
                    len_train,
                    n_steps_out,
                    val_target,
                    liste_features,
                    data,
                    model_type):
    start = len_train
    end = len(val_target) - n_steps_out
    vect_error_point = [0]*len(range(start, end + 1))
    vect_error_volume = [0]*len(range(start, end + 1))
    i = 0
    
    # On ajoute l'indicateur saisonier pour le covid
    x = np.arange(len(val_target))
    feature_saison = np.sin(x*np.pi/12.)
    feature_saison = feature_saison.reshape(len(val_target),1)
    
    # Scaling des inputs:
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target = scaler_target.fit(val_target.reshape(-1, 1))
    normalized_target = scaler_target.transform(val_target.reshape(-1, 1))
    normalized_target = normalized_target.reshape(len(val_target),1)
    
    Data_all_lisse = hstack((normalized_target ,feature_saison))
    # définition des scalers : 
    # On ajoute la liste des features une par une avec un scaling
    for i in range(len(liste_features)):
        # extraire la feauture 
        feature_ = data[liste_features[i]].values
        # scaler la feature
        scaler_ = MinMaxScaler(feature_range=(0, 1))
        scaler_ = scaler_.fit(feature_.reshape(-1, 1))
        feature_ = scaler_.transform(feature_.reshape(-1, 1))
        feature_ = feature_.reshape(len(val_target),1)
        # hstack avec les autres features
        Data_all_lisse = hstack((Data_all_lisse ,feature_))
        
    # + 1 pour la target, +1 pour la feature saison
    nb_features = len(liste_features) + 1 + 1
    n_input =  nb_features * look_back
    i = 0     
    print('ddddddddddddddddddddddans objectif volume')
    print(start)
    print(end + 1)
    for train_ in range(start, end + 1):
        print('ddddddddddddddddddddddans objectif volume')
        forecast_be_de = array(Data_all_lisse[:train_,:][-look_back:,:])
        if model_type == 'lstm':
            forecast_be_de = forecast_be_de.reshape(1,look_back,nb_features)
        if model_type == 'mlp':
            forecast_be_de = forecast_be_de.reshape(1,n_input)
        
        y_predict = model.predict(forecast_be_de, verbose=0)
        
        y_predict = scaler_target.inverse_transform(y_predict)
        y_predict = [item for sublist in y_predict for item in sublist]
        y = val_target[train_ :train_ + len(y_predict) ]
        
        vect_error_volume[i] = eval_error_volume(y, y_predict)
        i = i + 1
        
    
    moy = np.mean(np.array(vect_error_volume))
    sigma = np.std(np.array(vect_error_volume))
    #message = 'Score : ' +  str(moy) + ',  ecart type = ' + str(ET)
    #st.write(message)
    return moy, sigma

def deeplearning_objectif_mlp(space):
    print('deeplearning_objectif_mlpdeeplearning_objectif_mlpdeeplearning_objectif_mlp')
    n_steps_out = space['n_steps_out']
    train = space['train']
    look = space['look']
    epoch = space['epoch']
    batch_size = space['batch_size']
    regularisation = space['regularisation']
    data_fr_lisse = space['target']
    nb_features_random = space['nb_features_random']
    liste_features = space['liste_feature']
    print(liste_features)
    print(nb_features_random)
    # dans le cas où on veut les features en paramètres
    liste_features = random.sample(liste_features, k = nb_features_random)
    # on ordone la liste de features pour l'analyse
    liste_features.sort()
    optimizer = space['optimizer']
    nb_neurones_input = space['units1'] 
    nb_neurones_interm = [0 ,0]
    data = space['data']
    
    if space['choice']['layers']== 'two':
        nb_neurones_interm = [space['choice']['units2'], 0]
    
    if  space['choice']['layers']== 'three':
        nb_neurones_interm = [space['choice']['units2'], space['choice']['units3']]
    
    # On ajoute l'indicateur saisonier pour le covid
    x = np.arange(len(data_fr_lisse))
    feature_saison = np.sin(x*np.pi/12.)
    feature_saison = feature_saison.reshape(len(data_fr_lisse),1)
    
    
    
    # Scaling des inputs:
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target = scaler_target.fit(data_fr_lisse.reshape(-1, 1))
    normalized_target = scaler_target.transform(data_fr_lisse.reshape(-1, 1))
    data_fr_lisse = normalized_target.reshape(len(data_fr_lisse),1)
    
    Data_all_lisse = hstack((data_fr_lisse ,feature_saison))
    
    # On ajoute la liste des features une par une avec un scaling
    for i in range(len(liste_features)):
        # extraire la feauture 
        feature_ = data[liste_features[i]].values
        # scaler la feature
        scaler_ = MinMaxScaler(feature_range=(0, 1))
        scaler_ = scaler_.fit(feature_.reshape(-1, 1))
        feature_ = scaler_.transform(feature_.reshape(-1, 1))
        feature_ = normalized_target.reshape(len(data_fr_lisse),1)
        # hstack avec les autres features
        Data_all_lisse = hstack((Data_all_lisse ,feature_))
    
    Data_train = Data_all_lisse[:train,:]
    
    #Data_predict = Data_all_lisse[train,:]
    len_Data_train = len(Data_train)
    n_steps = look
    #n_steps_out = 29
    Data_train_SVF, Data_predict_SVF = supervi_format_multi_imputs_fr(Data_train,
                                                                      n_steps,
                                                                      n_steps_out)
  
    n_input = Data_train_SVF.shape[1] * Data_train_SVF.shape[2]
    #print(n_input)
    Data_train_SVF = Data_train_SVF.reshape((Data_train_SVF.shape[0], n_input))
    #print(Data_train_SVF)
    model = Sequential()
    model.add(Dense(nb_neurones_input, activation='relu', input_dim=n_input,
                    kernel_regularizer=regularisation))
    model.add(Dropout(rate = space['dropout1']))
    for i in range(len(nb_neurones_interm)):
        if nb_neurones_interm[i] != 0 :
            model.add(Dense(nb_neurones_interm[i], activation='relu',
                            input_dim=n_input,kernel_regularizer=regularisation))
            model.add(Dropout(space['dropout1']))
            
    #model.add(Dense(4))
    model.add(Dense(n_steps_out,kernel_regularizer=regularisation))
    model.compile(optimizer='adam', loss='mse')
   
    for i in range(len(Data_predict_SVF)):
        Data_predict_SVF[i] = Data_predict_SVF[i].tolist()
    
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20)
    result = model.fit(Data_train_SVF, Data_predict_SVF, epochs=epoch,
              verbose=0, batch_size = batch_size,
              validation_split=0.1,
              callbacks=[es,TqdmCallback(verbose=1)],
              shuffle = True)
    
    moy, sigma = objectif_volume(model = model,
                                      look_back = space['look'],
                                      len_train = space['train'],
                                      n_steps_out = space['n_steps_out'],
                                      val_target = space['target'],
                                      liste_features = liste_features,
                                      data = space['data'],
                                      model_type = "mlp")
    print('Best validation loss of epoch:', moy)
    
    return {'loss': moy, 'status': STATUS_OK, 'model': model, 'params': space,
            'data_frame' : { 'n_steps_out' : n_steps_out,
                            'train' : train,
                            'look' : look,
                            'dropout1' : space['dropout1'],
                            'batch_size' : batch_size,
                            'regularisation' : regularisation,
                            'nb_features' : nb_features_random,
                            'liste_features' : liste_features,
                            'nb_neurones_input' : nb_neurones_input,
                            'nb_neurones_interm' : nb_neurones_interm,
                            'erreur_volume' : moy,
                            'ecart_type' : sigma}}


def deeplearning_objectif_lstm(space):
    print('deeplearning_objectif_lstm deeplearning_objectif_lstm deeplearning_objectif_lstm deeplearning_objectif_lstm ')
    n_steps_out = space['n_steps_out']
    train = space['train']
    look = space['look']
    epoch = space['epoch']
    batch_size = space['batch_size']
    regularisation = space['regularisation']
    data_fr_lisse = space['target']
    nb_features_random = space['nb_features_random']
    liste_features = space['liste_feature']
    print(liste_features)
    print(nb_features_random)
    # dans le cas où on veut les features en paramètres
    liste_features = random.sample(liste_features, k = nb_features_random)
    # on ordone la liste de features pour l'analyse
    liste_features.sort()
    optimizer = space['optimizer']
    nb_neurones_input = space['units1'] 
    nb_neurones_interm = [0 ,0]
    data = space['data']
    
    if space['choice']['layers']== 'two':
        nb_neurones_interm = [space['choice']['units2'], 0]
    
    if  space['choice']['layers']== 'three':
        nb_neurones_interm = [space['choice']['units2'], space['choice']['units3']]
    
    # On ajoute l'indicateur saisonier pour le covid
    x = np.arange(len(data_fr_lisse))
    feature_saison = np.sin(x*np.pi/12.)
    feature_saison = feature_saison.reshape(len(data_fr_lisse),1)
    
    
    
    # Scaling des inputs:
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target = scaler_target.fit(data_fr_lisse.reshape(-1, 1))
    normalized_target = scaler_target.transform(data_fr_lisse.reshape(-1, 1))
    data_fr_lisse = normalized_target.reshape(len(data_fr_lisse),1)
    
    Data_all_lisse = hstack((data_fr_lisse ,feature_saison))
    
    # On ajoute la liste des features une par une avec un scaling
    for i in range(len(liste_features)):
        # extraire la feauture 
        feature_ = data[liste_features[i]].values
        # scaler la feature
        scaler_ = MinMaxScaler(feature_range=(0, 1))
        scaler_ = scaler_.fit(feature_.reshape(-1, 1))
        feature_ = scaler_.transform(feature_.reshape(-1, 1))
        feature_ = normalized_target.reshape(len(data_fr_lisse),1)
        # hstack avec les autres features
        Data_all_lisse = hstack((Data_all_lisse ,feature_))
    
    Data_train = Data_all_lisse[:train,:]
    
    #Data_predict = Data_all_lisse[train,:]
    len_Data_train = len(Data_train)
    n_steps = look
    #n_steps_out = 29
    Data_train_SVF, Data_predict_SVF = supervi_format_multi_imputs_fr(Data_train,
                                                                      n_steps,
                                                                      n_steps_out)
  
    #n_input = Data_train_SVF.shape[1] * Data_train_SVF.shape[2]
    #print(n_input)
    n_features = Data_train_SVF.shape[2]
    #Data_train_SVF = Data_train_SVF.reshape((Data_train_SVF.shape[0], n_input))
    #print(Data_train_SVF)
    model = Sequential()
    if nb_neurones_interm[0] == 0:
        return_sequences=False
    else:
        return_sequences=True
        
    model.add(LSTM(nb_neurones_input, activation='relu',
                   return_sequences=return_sequences,
                   input_shape=(look, n_features)))
    model.add(Dropout(rate = space['dropout1']))
    if nb_neurones_interm[0] != 0:
        model.add(LSTM(nb_neurones_interm[0], activation='relu'))
        model.add(Dropout(rate = space['dropout1']))
        
    if nb_neurones_interm[1] != 0:
        model.add(Dense(nb_neurones_interm[1], activation='relu',
                        kernel_regularizer=regularisation))
        model.add(Dropout(rate = space['dropout1']))
    
    model.add(Dense(n_steps_out,kernel_regularizer=regularisation))
 
    model.compile(optimizer='adam', loss='mse') 
    
    for i in range(len(Data_predict_SVF)):
        Data_predict_SVF[i] = Data_predict_SVF[i].tolist()
    
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=30)
    result = model.fit(Data_train_SVF, Data_predict_SVF, epochs=epoch,
              verbose=0, batch_size = batch_size,
              validation_split=0.1,
              callbacks=[es,TqdmCallback(verbose=1)],
              shuffle = True)
    
    moy, sigma = objectif_volume(model = model,
                                      look_back = space['look'],
                                      len_train = space['train'],
                                      n_steps_out = space['n_steps_out'],
                                      val_target = space['target'],
                                      liste_features = liste_features,
                                      data = space['data'],
                                      model_type = "lstm")
    print('Best validation loss of epoch:', moy)
    
    return {'loss': moy, 'status': STATUS_OK, 'model': model, 'params': space,
            'data_frame' : { 'n_steps_out' : n_steps_out,
                            'train' : train,
                            'look' : look,
                            'dropout1' : space['dropout1'],
                            'batch_size' : batch_size,
                            'regularisation' : regularisation,
                            'nb_features' : nb_features_random,
                            'liste_features' : liste_features,
                            'nb_neurones_input' : nb_neurones_input,
                            'nb_neurones_interm' : nb_neurones_interm,
                            'erreur_volume' : moy,
                            'ecart_type' : sigma}}
