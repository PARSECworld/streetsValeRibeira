import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Average 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input 
import tensorflow.keras

#from tqdm.notebook import tqdm
from tqdm import tqdm
from ipywidgets import IntProgress

import subprocess
import datetime

import pandas as pd
import numpy as np
import random
import re
import os

import matplotlib.pyplot as plt


VGG_SHAPE   = (4096,) #number of features out from VGG
NUM_FOLDS   = 5
NUM_ANGLES  = 4 #DIRS = [0, 90, 180, 270] for street images
NUM_CLASSES = 5  #scores = [1, 2, 3, 4, 5]
BATCH_SIZE = 400
EPOCHS = 1000
 

import customized_metrics

def Pearson(y_true, y_pred): 
    return customized_metrics.Pearson(y_true, y_pred, axis=-2)


def KendallTau(y_true, y_pred): 
    return customized_metrics.KendallTau(y_true, y_pred)


def create_model():
    '''
    ...
    '''
    inputs = [Input(shape=VGG_SHAPE) for _ in range(NUM_ANGLES)]
    
    l1 = []
    for i in inputs:
        x = Dense(512, activation='relu')(i)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        l1.append(x)
    
    l2 = Average()(l1)
    l3 = Dense(64, activation='relu')(l2)
    l3 = BatchNormalization()(l3) 
    output = Dense(NUM_CLASSES, activation='sigmoid')(l3) 
    
    model = Model(inputs, output)
    opt= tf.keras.optimizers.Adam(learning_rate=5e-6)    
    #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=opt,     
    loss=['categorical_crossentropy'],     
    metrics=['accuracy','mean_absolute_error', Pearson,KendallTau],
    run_eagerly=True
    ) 
#tf.keras.losses.MeanSquaredError() , tf.keras.losses.CategoricalCrossentropy()], 
    
    
    return model

def split_data():
    if True:  # to be run only once!
        locations = {}

        def get_fold(filename):        
            location = re.sub(r'\-\d+\.jpg', '', filename)
            if not location in locations:
                locations[location] = np.random.randint (0, NUM_FOLDS)
            return locations[location]

        df = pd.read_csv('geo.csv')
        df['fold'] = df['filename'].apply(get_fold)
        df.to_csv('geo_fold.csv', index=None)      


def data_generator(batch_size, epochs, fold, indicator, folder_indicator):
    '''
    ...
    '''
    print("fold:"+str(fold))
    csv_filename = f'{folder_indicator}/folds/{fold}.csv'

    idh = pd.read_csv('IDHMs/IDHM_ValeRibeira.csv', usecols=['Cod_setor', indicator])
    df = pd.read_csv('geo_fold.csv')
    df = pd.merge(df, idh, left_on='setor', right_on='Cod_setor', how='inner') 
    
    for epoch in range(epochs):   
        for chunk in pd.read_csv('vgg_features.csv', chunksize=4 * batch_size):                    

            chunk['filename'] = chunk['filename'].apply(lambda x: x.split('/')[2]) 
            #format ../images/Ur0pYB2-_GRdWdiDs7fLDw-0.jpg
            #split path into [2] folder / images / image*.jpg
            batch = pd.merge(df, chunk, left_on='filename', right_on='filename', how='inner')
            if batch.empty:
                continue
            test = batch[batch['fold'] == fold]
            
            if epoch == 0:
                mode = 'a' if os.path.exists(csv_filename) else 'w'                
                test.to_csv(csv_filename, index=None, mode=mode,columns = ['filename','setor','fold', indicator)
                #test.to_csv(csv_filename, index=None, mode=mode) #write all columns
                
            batch = batch[batch['fold'] != fold]
            #print(type(batch)) #(64, 4101)
            if len(batch) > 0:
                X = [batch.iloc[i::NUM_ANGLES][[f'v{i}' for i in range(VGG_SHAPE[0])]] for i in range(NUM_ANGLES)]  #imgs                
                y = tf.keras.utils.to_categorical(batch[indicator].iloc[0::NUM_ANGLES] - 1, NUM_CLASSES) #targets
 
                yield X, y


# BATCH_SIZE = 16
# EPOCHS = 3


def main(indicator, folder_indicator):
    '''
    ...
    '''
    for FOLD in tqdm(range(0, NUM_FOLDS)):

        model = create_model()
        
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=f"{folder_indicator}/logs/FOLD{FOLD}-{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}",
            histogram_freq=0, write_graph=False, write_images=False,
            update_freq='batch')
            
        savemodel = tf.keras.callbacks.ModelCheckpoint( 
            filepath=f"{folder_indicator}/models/FOLD{FOLD}-{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}",
            save_weights_only=False,
            monitor='loss',
            mode='min',
            verbose=1,
            save_best_only=True)
        
        datagen = data_generator(BATCH_SIZE, EPOCHS, FOLD, indicator, folder_indicator)
     
        history=model.fit(
            datagen,
            steps_per_epoch=112368//BATCH_SIZE//NUM_ANGLES,
            epochs=EPOCHS,
            callbacks=[tensorboard, savemodel],
            verbose=1
        )        
        # # list all data in history
        # print(history.history.keys())


        # save training history
        hist_csv_file = f'{folder_indicator}/history/history-FOLD{FOLD}.csv'
        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        del tensorboard
        del savemodel
        del datagen
        del model
    #    

    
if __name__ == "__main__":
    
    #python train.py    
    
    np.random.seed(28657) #seed random generator for reproduciblity
    
    if False:
        split_data()#to be run once only!
        

    #main(indicator='quintilAlfabetizacao',folder_indicator='literacy')
 
    main(indicator='quintilRenda',folder_indicator='income')
    
    
