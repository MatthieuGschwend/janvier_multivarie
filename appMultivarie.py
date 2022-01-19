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
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates


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
    #print(normalized_target)
    #print(forecast_be_de)
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
    
    
    scaler_target = MinMaxScaler(feature_range=(0, 1))
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


def plot_result(y_predict,val_target,len_train, index, titre):
    

    fig, axs = plt.subplots(figsize=(15,6))
    
    y_predict = [item for sublist in y_predict for item in sublist]
    x = np.arange(len(val_target))
    x_predict = np.arange(len_train, len_train  + len(y_predict))
    axs.plot(x, val_target,'o-', c = 'blue',label="target")
    axs.plot(x_predict,y_predict,'o-', c = 'green',label="Prédiction")
    
    axs.set_xticks(x,index)
    start, end = axs.get_xlim()
    axs.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.legend(bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=80, fontsize=15)
    plt.title(titre,fontsize=25)
    return fig

### back test

def eval_error_moy(y, y_pred):
    mean = np.mean(y)
    #st.write(mean)
    res = 0
    for i in range(len(y)):
        res = res + abs(y[i] - y_pred[i])/len(y)
    #st.write(y)
    #st.write(y_pred)

    return (100./mean)*res

def eval_error_volume(y, y_pred):
    sum_y = np.sum(y)
    sum_y_pred = np.sum(y_pred)
    return abs((sum_y-sum_y_pred)/sum_y)*100
        

def back_test(n_step_out,
              pourcentage_train,
              look_back,
              epoch,
              batch_size,
              regularisation,
              nb_neurones_input,
              nb_neurones_interm,
              val_target,
              val_covid,
              val_trend,
              data):
    
    pourcentage_train = pourcentage_train / 100
    start_train = int(len(val_target)*pourcentage_train)
    end_train = len(val_target) - n_step_out
    vect_error_point = [0]*len(range(start_train,end_train + 1))
    vect_error_volume = [0]*len(range(start_train,end_train + 1))
    i = 0
    for train_ in range(start_train,end_train + 1):
        #st.write(i)
        y_pred = deeplearning_lstm(n_step_out,
                                  train_,
                                  look_back,
                                  epoch,
                                  batch_size,
                                  regularisation,
                                  nb_neurones_input,
                                  nb_neurones_interm,
                                  val_target,
                                  val_covid,
                                  val_trend)
        y_pred = [item for sublist in y_pred for item in sublist]
        y = val_target[train_ :train_ + n_step_out ]
        vect_error_point[i] = eval_error_moy(y,y_pred)
        vect_error_volume[i] = eval_error_volume(y, y_pred)
        i = i+1
    
    index = data.index[start_train : end_train + 1 ]
    titre1 = ' Erreur moyenne en % point par point'
    titre2 = ' Erreur moyenne en % sur le volume'
    df_error = pd.DataFrame({titre2 : vect_error_volume,titre1 : vect_error_point},
                            index = index)
    
    
    st.dataframe(df_error)
    
    return vect_error_point, vect_error_volume
        

def interface():
    sns.set_theme(style="darkgrid")
    data = pd.read_csv('data_full_be_de_es.csv', sep = ',')
    data = data.set_index(data.columns[0])
    data.index = data.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
    
    list_col = data.columns
    #list_col = np.delete(list_col, 0)
    
    var_1 = st.sidebar.selectbox('Sélection de la série à predire',list_col)
    #var_2 = st.sidebar.selectbox('Première feature',list_col)
    #var_3 = st.sidebar.selectbox('Deuxième feature',list_col)
    type_multi = st.sidebar.radio("Ajouter des séries explicatives ?",
                          ('Non',
                           'Une série',
                           'Deux séries'))
    
    if type_multi == 'Deux séries':
        options = st.sidebar.multiselect(
         'Sélection de deux données explicatives',
         list_col,
         [list_col[3], list_col[6]])
        #st.write(options[0])
        var_2 = options[0]
        var_3 = options[1]
    
    if type_multi == 'Une série':
        options = st.sidebar.selectbox(
         'Sélection de deux données explicatives',
         list_col)
        var_2 = options
        var_3 = var_1
        
    if type_multi == 'Non':
        var_2 = var_1
        var_3 = var_1
    
    
    if st.sidebar.checkbox('Visualiser les données', value=False):
        st.pyplot(simple_visu(var_1, data, var_1))
        if type_multi == 'Une série':
            st.pyplot(simple_visu(var_2, data, var_2))
        if type_multi == 'Deux séries':
            st.pyplot(simple_visu(var_2, data, var_2))
            st.pyplot(simple_visu(var_3, data, var_3))
        

    nb_couches = 0
    neurones_couches = 0
    val_target = data[var_1].values
    val_covid = data[var_2].values
    val_trend = data[var_3].values

    type_model = st.radio("Quel modèle utiliser ?",
                          ('réseau de neurones classiques',
                           'réseau de neurones LSTM'))
    
   # Paramtres par défault 
    n_step_out  = 4
    len_Data_train = len(val_target)
    look_back = 20
    epoch =  1000
    batch_size = 50
    regularisation = 'l2'
    nb_neurones_input = 50
    nb_neurones_interm1 = 0
    nb_neurones_interm2 =  0

    if type_model == 'réseau de neurones LSTM':
        look_back = 5
        batch_size = 15
        nb_neurones_input = 50
        nb_neurones_interm1 = 0
        nb_neurones_interm2 = 5
        
    nb_neurones_interm = [nb_neurones_interm1, nb_neurones_interm2]
    
    
    # Paramètres avancés
    if st.sidebar.checkbox('Configuration avancée', value=False):
        col4, col5, col6 = st.columns(3)
        with col4:
            n_step_out = st.number_input('Nombre de pas de prévision (out)'
                                         ,value = 4)
            len_Data_train = st.number_input("Taille de l'entrainement",
                                             value = len(val_target))
            
            if type_model == 'réseau de neurones classiques':
                nb_look_back = 20
            else :
                nb_look_back = 5
                
            look_back = st.number_input('Nombre de pas en input (look back)'
                                        ,value = nb_look_back)
            
            
        with col5:
            epoch = st.number_input("Nombre d'entrainements (epoch)",
                                    value = 1000)
            if type_model == 'réseau de neurones classiques':
                nb_batch = 50
            else :
                nb_batch = 15
            batch_size = st.number_input("Taille des paquets (batch_size)",
                                         value = nb_batch)
            regularisation = st.selectbox('régularisation : ',
                                          ['l2','l1',None])
            
            
        if type_model == 'réseau de neurones classiques':
            with col6:
                nb_neurones_input = st.number_input("Neurones input layer",
                                                    value = 50)
                nb_neurones_interm1 = st.number_input("Neurones layer 2",
                                                      value = 0)
                nb_neurones_interm2 = st.number_input("Neurones layer 3",
                                                      value = 0)
                nb_neurones_interm = [nb_neurones_interm1, nb_neurones_interm2]
        
        if type_model == 'réseau de neurones LSTM':
            with col6:
                nb_neurones_input = st.number_input("Neurones input layer",
                                                    value = 50)
                nb_neurones_interm1 = st.number_input("Neurones layer 2 LSTM",
                                                      value = 0)
                nb_neurones_interm2 = st.number_input("Neurones layer classique",
                                                      value = 5)
                nb_neurones_interm = [nb_neurones_interm1, nb_neurones_interm2]
        
        # Les définitions
        if st.checkbox('Définitions', value=False):
            with col4:
                st.markdown("""* **Nombre de pas de prévision**
                            Une prévision dans le futur selon le nombre
                            de pas de temps voulu. Ici un pas de temps représente
                            2 semaines.""")
                
                st.markdown("""* **Taille de l'entrainement**   
                            On peut choisir d'entrainer le modèle seulement 
                            sur une partie des données et comparer ainsi la 
                            prévision avec les observations réelles. La prévision
                            se réalise donc à la suite du jeu d'entainement.""")
               
                st.markdown("""* **Nombre de pas en input**   
                            Le modèle prend en entrée (input) un nombre de
                            pas de temps dans le passé pour prédire le futur. """)
                            
            with col5:
                st.markdown("""* **Nombre d'entrainements**   
                            Le modèle doit s'entrainer à de multiples reprises
                            afin d'être efficace. Augmenter le nombre d'entrainements
                            entraîne un plus gros coût de calcul mais assure une
                            meilleure convergence""")
                
                st.markdown("""* **Taille des paquets**  
                            Un entrainement se fait par paquet de données. Si 
                            ces paquets sont grands alors l'entrainement est rapide
                            mais la convergence n'est pas toujours obtenue. Si les 
                            paquets sont petits l'entrainement est plus long mais
                            plus précis """)
                
                st.markdown("""* **Regularisation**   
                            D'un entainement à l'autre les résultats peuvent varier
                            du fait de la nature stochastique du modèle. Ajouter un
                            terme de régularisation stabilise donc cette variance.""")
            with col6:
                st.markdown("""* **Neurones input layer**   
                            Un réseau est composé de plusieurs couches, chacune des
                            couches comporte un nombre de neurones. Plus le nombre
                            de neurones est grand plus plus le modèle devient capable
                            apprendre des schémas complexes. Attention : il faut 
                            néanmoins adapter cette complexité à celle de notre 
                            problème.""")
                            
                st.markdown("""* **Couches supplémentaires**   
                            Un modèle peut contenir plusiers couches de différentes
                            natures, avec un nombre de neurones associé. Lorsque 
                            le nombre de neuronnes est à 0 (défault) aucune couche n'est
                            ajoutée.""")

        
    #len_Data_train = len(val_target) - n_step_out
    #st.write(len(val_target) - n_step_out)
    #st.write(nb_neurones_interm)
    
    forcast = st.button('Réaliser la prévision')
    if forcast:
        with st.spinner('Entrainement en cours (estimation 45 sec)'):

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
        st.success('Done!')
        st.pyplot(plot_result(y_predict,val_target,len_Data_train,data.index,var_1))
    
    back_test_button = st.checkbox('Réaliser un back test')
    if back_test_button:
        pourcentage_train = st.number_input("Poucentage du dataset utilisé pour débuter le back test",
                                            value = 80)
        
        back_test_start = st.button('lancer le back test')
        if back_test_start:
            y_back_test = back_test(n_step_out,
                                    pourcentage_train,
                                    look_back,
                                    epoch,
                                    batch_size,
                                    regularisation,
                                    nb_neurones_input,
                                    nb_neurones_interm,
                                    val_target,
                                    val_covid,
                                    val_trend,
                                    data)
            
            #index_error = data.index[train_ + 1:train_ + 1 + n_step_out ]
            
        
        
    
    #st.dataframe(data)
interface()
# export_ppt(generation_generique=False)

