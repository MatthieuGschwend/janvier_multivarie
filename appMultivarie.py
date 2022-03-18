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
import json
from numpy import array
from numpy import hstack
#from keras.models import Model
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
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import holidays
import hyperopt_mod as hm
import arima_mod as am
import random


#%% Fonctions de base
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


def simple_visu(y,data,titre):
    fig, axs = plt.subplots(figsize=(20,3))
    sns.set_theme(style="darkgrid")
    plt.xticks(rotation=80, fontsize=15)
    plt.title(titre,fontsize=25)
    sns.lineplot(x=data.index, y=y,
                 data=data)
    return fig


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

# %% Partie modèles
def deeplearning_mlp(n_steps_out,
                     train,
                     look,
                     epoch,
                     batch_size,
                     regularisation,
                     nb_neurones_input,
                     nb_neurones_interm,
                     data_fr_lisse,
                     liste_features,
                     data,
                     sinus = True):
    
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
        feature_ = feature_.reshape(len(data_fr_lisse),1)
        # hstack avec les autres features
        Data_all_lisse = hstack((Data_all_lisse ,feature_))
    
    Data_train = Data_all_lisse[:train,:]
    st.write(Data_all_lisse)
    print(Data_all_lisse)
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
    model.add(Dropout(rate = 0.1))
    for i in range(len(nb_neurones_interm)):
        if nb_neurones_interm[i] != 0 :
            model.add(Dense(nb_neurones_interm[i], activation='relu',
                            input_dim=n_input,kernel_regularizer=regularisation))
    #model.add(Dense(4))
    model.add(Dense(n_steps_out,kernel_regularizer=regularisation))
    model.compile(optimizer='adam', loss='mse')
   
    for i in range(len(Data_predict_SVF)):
        Data_predict_SVF[i] = Data_predict_SVF[i].tolist()
    
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=50)
    model.fit(Data_train_SVF, Data_predict_SVF, epochs=epoch,
              verbose=0, batch_size = batch_size),
              #validation_split=0.1,
              #callbacks=[es,TqdmCallback(verbose=1)],
              #shuffle = True)
    forecast_be_de = array(Data_all_lisse[:len_Data_train,:][-n_steps:,:])
    forecast_be_de = forecast_be_de.reshape(1,n_input)
    #print(forecast_be_de)
    y_predict = model.predict(forecast_be_de, verbose=0)
    y_predict = scaler_target.inverse_transform(y_predict)
    
    return y_predict

def deeplearning_mlp_hp(params):
    
    target = params['target']
    feature_1 = params['feature_1']
    feature_2 = params['feature_2']
    
    scaler_target = MinMaxScaler(feature_range=(0, 1))
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



def deeplearning_lstm(n_steps_out,
                        train,
                        look,
                        epoch,
                        batch_size,
                        regularisation,
                        nb_neurones_input,
                        nb_neurones_interm,
                        data_fr_lisse,
                        liste_features,
                        data,
                        sinus = True):
            
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
        feature_ = feature_.reshape(len(data_fr_lisse),1)
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
    model.fit(Data_train_SVF, Data_predict_SVF, epochs=epoch, verbose=0, batch_size = batch_size)
    forecast_be_de = array(Data_all_lisse[:len_Data_train,:][-n_steps:,:])
    forecast_be_de = forecast_be_de.reshape(1,look,n_features)
    y_predict = model.predict(forecast_be_de, verbose=0)
    
    y_predict = scaler_target.inverse_transform(y_predict)
    
    return y_predict
#%% Back tests

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
        # petit debuggage : 
        print('petit debuggage')
        if vect_error_point[i] == vect_error_volume[i] : 
            print(vect_error_point[i])
            print(y)
            print(y_pred)
            res = 0
            for i in range(len(y)):
                res = res + abs(y[i] - y_pred[i])/len(y)
                print(res)
        i = i+1
    
    index = data.index[start_train : end_train + 1 ]
    titre1 = ' Erreur moyenne en % point par point'
    titre2 = ' Erreur moyenne en % sur le volume'
    df_error = pd.DataFrame({titre2 : vect_error_volume,titre1 : vect_error_point},
                            index = index)
    
    
    st.dataframe(df_error)
    
    return vect_error_point, vect_error_volume

def back_test_model(model,
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
    for train_ in range(start, end + 1):
        forecast_be_de = array(Data_all_lisse[:train_,:][-look_back:,:])
        if model_type == 'lstm':
            forecast_be_de = forecast_be_de.reshape(1,look_back,nb_features)
        if model_type == 'mlp':
            forecast_be_de = forecast_be_de.reshape(1,n_input)
        y_predict = model.predict(forecast_be_de, verbose=0)
        
        y_predict = scaler_target.inverse_transform(y_predict)
        y_predict = [item for sublist in y_predict for item in sublist]
        y = val_target[train_ :train_ + len(y_predict) ]
        vect_error_point[i] = eval_error_moy(y,y_predict)
        vect_error_volume[i] = eval_error_volume(y, y_predict)
        i = i + 1
    index = data.index[start : end + 1 ]
    titre1 = ' Erreur moyenne en % point par point'
    titre2 = ' Erreur moyenne en % sur le volume'
    df_error = pd.DataFrame({titre2 : vect_error_volume,titre1 : vect_error_point},
                            index = index)
    
    
    st.dataframe(df_error)
    st.write(vect_error_volume)
    st.write('test')
    st.write(np.array(vect_error_volume))
    moy = np.mean(np.array(vect_error_volume))
    ET = np.std(np.array(vect_error_volume))
    message = 'Score : ' +  str(moy) + ',  ecart type = ' + str(ET)
    st.write(message)
    return vect_error_point, vect_error_volume


#%% Hyperopt        

#%% interface

def interface():
    sns.set_theme(style="darkgrid")
    data = pd.read_csv('data_full.csv', sep = ',')
    data = data.set_index(data.columns[0])
    data[data < 0] = 0
    data.index = data.index.map(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
    data.index.names = ['Date']
    data.rename(columns={"Res tourisme Nice Allemagne":
                         "Résidence de tourisme Nice Allemagne",
                         "Res tourisme Nice Belgique":
                         "Résidence de tourisme Nice Belgique",
                         "Res tourisme Nice Espagne":
                         "Résidence de tourisme Nice Espagne"},
                         inplace = True)
    
    # 
    vancances_fr =  []
    vancances_es =  []
    vancances_be =  []
    vancances_de =  []

    for i in data.index:
        if i in holidays.France():
            vancances_fr.append(1)
        else:
            vancances_fr.append(0)
            
        if i in holidays.Spain():
            vancances_es.append(1)
        else:
            vancances_es.append(0)
            
        if i in holidays.Belgium():
            vancances_be.append(1)
        else:
            vancances_be.append(0)
            
        if i in holidays.Germany():
            vancances_de.append(1)
        else:
            vancances_de.append(0)

    data['Vacances France'] = vancances_fr
    data['Vacances Espagne'] = vancances_es
    data['Vacances Belgique'] = vancances_be
    data['Vacances Allemagne'] = vancances_de
    
    list_col = data.columns
    
    var_1 = st.sidebar.selectbox('Sélection de la série à prédire',list_col)
  

    
  
    options = st.sidebar.multiselect(
     'Sélection des données explicatives',
     list_col)

    # on assigne la target et le nom des features
    val_target = data[var_1].values
    feature_var = options
    
    if st.sidebar.checkbox('Visualiser les données', value=False):
        st.pyplot(simple_visu(var_1, data, var_1))
        for i in range(len(feature_var)):
            st.pyplot(simple_visu(feature_var[i], data, feature_var[i]))
            

    type_model = st.radio("Quel modèle utiliser ?",
                          ('réseau de neurones classiques',
                           'réseau de neurones LSTM'))
    if type_model == 'réseau de neurones classiques':
        current_model = 'mlp'
    elif type_model == 'réseau de neurones LSTM':
        current_model = 'lstm'
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
                            d'apprendre des schémas complexes. Attention : il faut 
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
                                             feature_var,
                                             data,
                                             sinus = True)

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
                                             feature_var,
                                             data,
                                             sinus = True)
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
                                    feature_var,
                                    data)
    
    hyperopt_search = st.checkbox('Trouver les meilleurs paramètres pour une série')
    if hyperopt_search:
        n_step_out = st.number_input("Selection du nombre de pas de prévision",
                                     value = 4)
        len_train  = st.number_input("Taille du jeu d'entainement",
                                     value = 91)
        
        space = {'n_steps_out' : n_step_out,
                 'train' : len_train,
                 'look' : scope.int(hp.quniform('look', 5, 25, 5)),
                 'epoch' : 2000,
                 'batch_size' : scope.int(hp.quniform('batch_size',5,91,15)),
                 'regularisation' : hp.choice('regularisation',['l1','l2',None]),
                 'nb_neurones_input' : scope.int(hp.quniform('units',10,100,5)),
                 'target' : val_target,
                 'liste_feature' : feature_var,
                 'data' : data  
        }
        #space = hm.build_space(len_train, n_step_out, val_target, feature_var, data)
        space = hm.build_space_features(len_train, n_step_out, val_target, feature_var, data)
        if  st.button('Réaliser la recherche des paramètres'):
            trials = Trials()
            if current_model == 'mlp':
                best = fmin(hm.deeplearning_mlp,
                            space,
                            algo=tpe.suggest,
                            max_evals=50,
                            trials=trials)
            if current_model == "lstm":
                best = fmin(hm.deeplearning_lstm,
                            space,
                            algo=tpe.suggest,
                            max_evals=15,
                            trials=trials)
            
            best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
            best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
            worst_model = trials.results[np.argmax([r['loss'] for r in trials.results])]['model']
            worst_params = trials.results[np.argmax([r['loss'] for r in trials.results])]['params']
            best_model.save('my_model_3_86.h5')

            for i in best_params.keys():
                if i not in ['target', 'feature_1','feature_2']:
                    st.write(i + ' : ') 
                    st.write(best_params[i]) 
        
        if  st.button('Réaliser la recherche des paramètres avec objectif'):
            trials = Trials()
            if current_model == 'mlp':
                best = fmin(hm.deeplearning_objectif_mlp,
                            space,
                            algo=tpe.suggest,
                            max_evals=50,
                            trials=trials)
            if current_model == "lstm":
                best = fmin(hm.deeplearning_objectif_lstm,
                            space,
                            algo=tpe.suggest,
                            max_evals=1,
                            trials=trials)
            
            best_model = trials.results[np.argmin([r['loss'] for r in trials.results])]['model']
            best_params = trials.results[np.argmin([r['loss'] for r in trials.results])]['params']
            worst_model = trials.results[np.argmax([r['loss'] for r in trials.results])]['model']
            worst_params = trials.results[np.argmax([r['loss'] for r in trials.results])]['params']
            best_model.save('my_model_3_86.h5')
            n_steps_out = []
            train = []
            look = []
            batch_size = []
            regularisation = []
            nb_features = []
            liste_features = []
            nb_neurones_input = []
            nb_neurones_interm = []
            erreur_volume = []
            ecart_type = []
            dropout1 = []


            for i in range(len(trials.results)):
                json_loc = trials.results[i]['data_frame']
                n_steps_out.append(json_loc['n_steps_out'])
                train.append(json_loc['train'])
                look.append(json_loc['look'])
                batch_size.append(json_loc['batch_size'])
                regularisation.append(json_loc['regularisation'])
                nb_features.append(json_loc['nb_features'])
                liste_features.append(json_loc['liste_features'])
                nb_neurones_input.append(json_loc['nb_neurones_input'])
                nb_neurones_interm.append(json_loc['nb_neurones_interm'])
                erreur_volume.append(json_loc['erreur_volume'])
                ecart_type.append(json_loc['ecart_type'])
                dropout1.append(json_loc['dropout1'])

            
            to_data_frame ={'n_steps_out' : n_steps_out,
                            'train' : train,
                            'look' : look,
                            'dropout1' : dropout1,
                            'batch_size' : batch_size,
                            'regularisation' : regularisation,
                            'nb_features' : nb_features,
                            'liste_features' : liste_features,
                            'nb_neurones_input' : nb_neurones_input,
                            'nb_neurones_interm' : nb_neurones_interm,
                            'erreur_volume' : erreur_volume,
                            'ecart_type' : ecart_type
                            }
            resultats_hyper_opt = pd.DataFrame(data=to_data_frame)
            resultats_hyper_opt.to_csv("sofiane_be_2_85_std.csv") 
                
                
            
            for i in best_params.keys():
                if i not in ['target', 'feature_1','feature_2']:
                    st.write(i + ' : ') 
                    st.write(best_params[i]) 

        
        if st.button('Meilleurs paramètres SARIMA (univarié)'):
            trials = Trials()
            space = am.build_space(len_train, n_step_out,val_target)
            best = fmin(am.deeplearning_objectif_sarima,
                        space,
                        algo=tpe.suggest,
                        max_evals=5,
                        trials=trials)
            
            n_steps_out = []
            train = []
            look = []
            p = []
            d = []
            q = []
            P = []
            D = []
            Q = []
            m = []
            t = []
            ecart_type = []
            erreur = []

            for i in range(len(trials.results)):
                json_loc = trials.results[i]['data_frame']
                n_steps_out.append(json_loc['n_steps_out'])
                train.append(json_loc['train'])
                p.append(json_loc['p'])
                d.append(json_loc['regularisation'])
                q.append(json_loc['nb_features'])
                P.append(json_loc['liste_features'])
                D.append(json_loc['nb_neurones_input'])
                Q.append(json_loc['nb_neurones_interm'])
                m.append(json_loc['erreur_volume'])
                t.append(json_loc['ecart_type'])
                ecart_type.append(json_loc['ecart_type'])
                erreur.append(json_loc['erreur_volume'])
            
            to_data_frame ={'n_steps_out' : n_steps_out,
                            'train' : train,
                            'p' : p,
                            'd' : d,
                            'q' : q,
                            'P' : P,
                            'D' : D,
                            'Q' : Q,
                            'm' : m,
                            't' : t,
                            'erreur_volume' : erreur,
                            'ecart_type' : ecart_type
                            }
            resultats_hyper_opt = pd.DataFrame(data=to_data_frame)
            resultats_hyper_opt.to_csv("sarima_be_2_85_std.csv") 
            
                #index_error = data.index[train_ + 1:train_ + 1 + n_step_out ]
        ## test de réutilisation du modèle
        if st.button('diagram create'):
            model = load_model('my_model_3_86.h5')
            #plot_model(model, to_file='my_model.png', show_shapes=True, show_layer_names=True)
            model_json = model.to_json()
            model.summary()
            print('New test')
            print(model_json)
            model_json = json.loads(model_json)
            #print(model_json["config"]['layers']['config']['batch_input_shape'])
            print(type(model_json["config"]['layers'][0]['config']['batch_input_shape'][1]))
           
        if st.button('back test du best model') : 
            print('NNNNNNNNNNNNNNNNNEEEEEEEEEEEEEEWWWWWWWWWWW')
            model = load_model('my_model_3_86.h5')
            model_json = model.to_json()
            model_json = json.loads(model_json)
            print(model_json)
            if current_model == 'mlp':
                look_back = model_json["config"]['layers'][0]['config']\
                    ['batch_input_shape'][1] / (len(feature_var) + 1 +1)
            if current_model == 'lstm':
                look_back = model_json["config"]['layers'][0]['config']\
                    ['batch_input_shape'][1]
            print('LE LOOK BACK')
            print(look_back)
            look_back = int(look_back)
            st.write(look_back)

            y_back_test = back_test_model(model,
                                          look_back,
                                          len_train,
                                          n_step_out,
                                          val_target,
                                          feature_var,
                                          data,
                                          model_type = current_model)
            
        
        
            
        
        
    
    #st.dataframe(data)
interface()
# export_ppt(generation_generique=False)

