import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras

#from tqdm.notebook import tqdm
from tqdm import tqdm
from ipywidgets import IntProgress

import subprocess
import datetime


import pandas as pd
import numpy as np
import re
import os
import glob
import sys

from customized_metrics import Pearson, KendallTau


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

 
# BATCH_SIZE = 16
# EPOCHS = 3


def main(indicator, folder_indicator):
    '''
    ...
    '''

    #to print headers of testing metrics
    sys.stdout=open("testing/testing_metrics.csv",'w')
    print("Fold\tLoss\tAccuracy\tMAE\tPearson\tKendallTau")    
    sys.stdout.close()

    for FOLD in tqdm(range(0, NUM_FOLDS)):    
        # Define per-fold score containers    
        loss_per_fold = []
        acc_per_fold = [] 
        MAE_per_fold = []
        pearson_per_fold = [] 
        kendall_per_fold = []    

        model = load_model(glob.glob(f'{folder_indicator}/models/FOLD{FOLD}*')[0],compile=True, custom_objects={'Pearson': Pearson,'KendallTau': KendallTau})
    
        csv_file = f'{folder_indicator}/folds/fold-{FOLD}.csv'
        

        #for i, batch in enumerate(tqdm(pd.read_csv(csv_file, chunksize=100*NUM_ANGLES))):
        for i, batch in enumerate(tqdm(pd.read_csv(csv_file, chunksize=32*NUM_ANGLES))):
            batch = batch[batch['fold'] == FOLD]
            batch['location'] = batch['filename'].apply(lambda x: re.sub('\-\d+\.jpg', '', x))
            location_value_counts = batch['location'].value_counts()
            batch['total'] = batch['location'].map(location_value_counts)
            batch = batch[batch['total'] == NUM_ANGLES]
            if len(batch) == 0:
                continue
 
            if len(batch) > 0:
                X = [batch.iloc[i::NUM_ANGLES][[f'v{i}' for i in range(VGG_SHAPE[0])]] for i in range(NUM_ANGLES)]  #imgs                
                y = tf.keras.utils.to_categorical(batch[indicator].iloc[0::NUM_ANGLES] - 1, NUM_CLASSES) #targets

            
            del batch
            # Generate generalization metrics
            ##['loss', 'accuracy', 'mean_absolute_error', 'Pearson', 'KendallTau']            
            scores = model.evaluate(X,y, verbose=0)
            #print(model.metrics_names)
            
            loss_per_fold.append(scores[0])
            acc_per_fold.append(scores[1])
            MAE_per_fold.append(scores[2])
            pearson_per_fold.append(scores[3]) 
            kendall_per_fold.append(scores[4])

        del model      


  
        #save data from testing
        test_df=pd.DataFrame({'loss':loss_per_fold, 'accuracy':acc_per_fold, 'mean_absolute_error':MAE_per_fold, 'Pearson':pearson_per_fold, 'KendallTau':kendall_per_fold}) 
        test_csv_file = f'{folder_indicator}/testing/test-{FOLD}.csv'    
        with open(test_csv_file, mode='w') as f:
            test_df.to_csv(f)
        #    
        #metrics=['loss',accuracy','mean_absolute_error', Pearson,KendallTau],
        # == Provide average scores ==

        sys.stdout=open("testing/testing_metrics.csv",'a')
        ##print("Fold\tLoss\tAccuracy\tMAE\tPearson\tKendallTau")        
        print(str(np.mean(loss_per_fold))+"\t" +
         str(np.mean(acc_per_fold)) + "\t" + 
         str(np.mean(MAE_per_fold)) + "\t" + 
         str(np.mean(pearson_per_fold)) + "\t" + 
         str(np.mean(kendall_per_fold)))
        sys.stdout.close()
    
if __name__ == "__main__":
    
    #Testing 
    
    np.random.seed(28657) #seed random generator for reproduciblity
 
    #main(indicator='quintilAlfabetizacao',folder_indicator='literacy')
    
    main(indicator='quintilRenda',folder_indicator='income')
    
    
