# -*- coding: utf-8 -*-
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


def supervi_format_multi_imputs_fr(donnes_train, n_steps, n_steps_out):
    X_train, y_predict = list(), list()
    for i in range(len(donnes_train)):
        # on cherche l'index de fin
        end_ix = i + n_steps
        out_end_ix = end_ix + n_steps_out-1

        # condition d'arrêt de la boucle : sinon valeurs NA 
        if out_end_ix > len(donnes_train):
            break
        seq_x, seq_y = donnes_train[i:end_ix,:], donnes_train[end_ix-1:out_end_ix,0]
        X_train.append(seq_x)
        y_predict.append(seq_y)
    return array(X_train), array(y_predict)


def simple_visu(y,data,titre):
    fig, axs = plt.subplots(figsize=(20,3))
    sns.set_theme(style="darkgrid")
    plt.xticks(rotation=80, fontsize=15)
    plt.title(titre,fontsize=25)
    sns.lineplot(x=data.index, y=y,
                 data=data)
    return fig


def deeplearning_mlp(n_steps_out,
                     train,
                     look,
                     epoch,
                     batch_size,
                     regularisation,
                     nb_neurones_input,
                     nb_neurones_interm,
                     data_fr_lisse,
                     data_be_lisse,
                     data_de_lisse):
    
    
    scaler_target = MinMaxScaler(feature_range=(-5, 5))
    scaler_covid = MinMaxScaler(feature_range=(0, 1))
    scaler_trend = MinMaxScaler(feature_range=(0, 1))
    
    scaler_target = scaler_target.fit(data_fr_lisse.reshape(-1, 1))
    scaler_covid = scaler_covid.fit(data_be_lisse.reshape(-1, 1))
    scaler_trend = scaler_trend.fit(data_de_lisse.reshape(-1, 1))
    
    normalized_target = scaler_target.transform(data_fr_lisse.reshape(-1, 1))
    normalized_covid = scaler_covid.transform(data_be_lisse.reshape(-1, 1))
    normalized_trend = scaler_trend.transform(data_de_lisse.reshape(-1, 1))
    
    
    data_fr_lisse = normalized_target.reshape(len(data_fr_lisse),1)
    data_be_lisse = normalized_covid.reshape(len(data_fr_lisse),1)
    data_de_lisse = normalized_trend.reshape(len(data_fr_lisse),1)


    Data_all_lisse = hstack((data_fr_lisse, data_be_lisse, data_de_lisse))
    
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
    model.add(Dense(50, activation='relu', input_dim=n_input,
                    kernel_regularizer=regularisation))
    for i in range(len(nb_neurones_interm)):
        if nb_neurones_interm[i] != 0 :
            model.add(Dense(nb_neurones_interm[i], activation='relu',
                            input_dim=n_input,kernel_regularizer=regularisation))
    #model.add(Dense(4))
    model.add(Dense(n_steps_out,kernel_regularizer=regularisation))
    model.compile(optimizer='adam', loss='mse')
   
    for i in range(len(Data_predict_SVF)):
        Data_predict_SVF[i] = Data_predict_SVF[i].tolist()

    model.fit(Data_train_SVF, Data_predict_SVF, epochs=epoch, verbose=0, batch_size = batch_size)
    forecast_be_de = array(Data_all_lisse[:len_Data_train,:][-n_steps:,:])
    #print(forecast_be_de)
    #print(Data_all_lisse)
    #print(forecast_be_de)
    forecast_be_de = forecast_be_de.reshape(1,n_input)
    #print(forecast_be_de)
    y_predict = model.predict(forecast_be_de, verbose=0)
    
    y_predict = scaler_target.inverse_transform(y_predict)
    
    return y_predict


def deeplearning_lstm(n_steps_out,
                      train,
                      look,
                      epoch,
                      batch_size,
                      regularisation,
                      nb_neurones_input,
                      nb_neurones_interm,
                      data_fr_lisse,
                      data_be_lisse,
                      data_de_lisse):
    
    
    scaler_target = MinMaxScaler(feature_range=(-5, 5))
    scaler_covid = MinMaxScaler(feature_range=(0, 1))
    scaler_trend = MinMaxScaler(feature_range=(0, 1))
    
    scaler_target = scaler_target.fit(data_fr_lisse.reshape(-1, 1))
    scaler_covid = scaler_covid.fit(data_be_lisse.reshape(-1, 1))
    scaler_trend = scaler_trend.fit(data_de_lisse.reshape(-1, 1))
    
    normalized_target = scaler_target.transform(data_fr_lisse.reshape(-1, 1))
    normalized_covid = scaler_covid.transform(data_be_lisse.reshape(-1, 1))
    normalized_trend = scaler_trend.transform(data_de_lisse.reshape(-1, 1))
    
    
    data_fr_lisse = normalized_target.reshape(len(data_fr_lisse),1)
    data_be_lisse = normalized_covid.reshape(len(data_fr_lisse),1)
    data_de_lisse = normalized_trend.reshape(len(data_fr_lisse),1)


    Data_all_lisse = hstack((data_fr_lisse, data_be_lisse, data_de_lisse))
    
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
    model.fit(Data_train_SVF, Data_predict_SVF, epochs=epoch, verbose=0, batch_size = batch_size)
    forecast_be_de = array(Data_all_lisse[:len_Data_train,:][-n_steps:,:])
    forecast_be_de = forecast_be_de.reshape(1,look,n_features)
    y_predict = model.predict(forecast_be_de, verbose=0)
    
    y_predict = scaler_target.inverse_transform(y_predict)
    
    return y_predict


def plot_result(y_predict,val_target,len_train):
    

    fig, axs = plt.subplots(figsize=(15,6))
    
    y_predict = [item for sublist in y_predict for item in sublist]
   
    x = np.arange(len(val_target))
    x_predict = np.arange(len_train, len_train  + len(y_predict))
    axs.plot(x, val_target,'o-', c = 'blue',label="target")
    axs.plot(x_predict,y_predict,'o-', c = 'green',label="Prédiction")
    plt.legend(bbox_to_anchor=(1.15, 1))
    return fig
    


def interface():
    sns.set_theme(style="darkgrid")
    data = pd.read_csv('data_full_be_de_es.csv', sep = ',')
    data = data.set_index(data.columns[0])
    data.index = data.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
    
    list_col = data.columns
    #list_col = np.delete(list_col, 0)
    
    var_1 = st.sidebar.selectbox('Selection de la série à predire',list_col)
    #var_2 = st.sidebar.selectbox('Première feature',list_col)
    #var_3 = st.sidebar.selectbox('Deuxième feature',list_col)
    
    options = st.sidebar.multiselect(
     'Selectionner deux features',
     list_col,
     [list_col[3], list_col[6]])
    st.write(options[0])
    var_2 = options[0]
    var_3 = options[1]
    
    
    if st.sidebar.checkbox('Visualiser les données', value=False):
        st.write('test')
        st.pyplot(simple_visu(var_1, data, var_1))
        st.pyplot(simple_visu(var_2, data, var_2))
        st.pyplot(simple_visu( var_3, data, var_3))
        
    

    nb_couches = 0
    neurones_couches = 0
    val_target = data[var_1].values
    val_covid = data[var_2].values
    val_trend = data[var_3].values

    type_model = st.radio("Quel modèle utiliser ?",
                          ('réseau de neurones classiques',
                           'réseau de neurones LSTM'))

    st.write('Selection des paramètres')
    col4, col5, col6 = st.columns(3)
    
    with col4:
        n_step_out = st.number_input('Nombre de pas de prévision (out)', value = 4)
        len_Data_train = st.number_input("Taille de l'entrainement", value = len(val_target))
        
        look_back = st.number_input('Nombre de pas en input (look back)', value = 20)
        
        
    with col5:
        epoch = st.number_input("Nombre d'entrainements (epoch)", value = 1000)
        batch_size = st.number_input("Taille des paquets (batch_size)", value = 50)
        regularisation = st.selectbox('Regularisation : ',['l2','l1',None])
        
        
    if type_model == 'réseau de neurones classiques':
        with col6:
            nb_neurones_input = st.number_input("Neurones input layer", value = 50)
            nb_neurones_interm1 = st.number_input("Neurones layer 2", value = 0)
            nb_neurones_interm2 = st.number_input("Neurones layer 3", value = 0)
            nb_neurones_interm = [nb_neurones_interm1, nb_neurones_interm2]
    
    if type_model == 'réseau de neurones LSTM':
        with col6:
            nb_neurones_input = st.number_input("Neurones input layer", value = 50)
            nb_neurones_interm1 = st.number_input("Neurones layer 2 LSTM", value = 0)
            nb_neurones_interm2 = st.number_input("Neurones layer classique", value = 5)
            nb_neurones_interm = [nb_neurones_interm1, nb_neurones_interm2]
    
        
    #len_Data_train = len(val_target) - n_step_out
    st.write(len(val_target) - n_step_out)
    
    forcast = st.button('Réaliser la prévision')
    if forcast:
        if type_model == 'réseau de neurones classiques':
            y_predict = deeplearning_mlp(n_step_out,
                                         len_Data_train,
                                         look_back,
                                         epoch,
                                         batch_size,
                                         regularisation,
                                         nb_neurones_input,
                                         nb_neurones_interm,
                                         val_target,
                                         val_covid,
                                         val_trend)
        if type_model == 'réseau de neurones LSTM':
            y_predict = deeplearning_lstm(n_step_out,
                                         len_Data_train,
                                         look_back,
                                         epoch,
                                         batch_size,
                                         regularisation,
                                         nb_neurones_input,
                                         nb_neurones_interm,
                                         val_target,
                                         val_covid,
                                         val_trend)
        
    
        st.pyplot(plot_result(y_predict,val_target,len_Data_train))
    
    #st.dataframe(data)
interface()
# export_ppt(generation_generique=False)

