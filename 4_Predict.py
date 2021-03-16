from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tqdm.notebook import tqdm
from ipywidgets import IntProgress

import pandas as pd
import numpy as np
import re
import os
import glob



VGG_SHAPE  = (4096,) #number of features out from VGG
NUM_FOLDS  = 5
NUM_ANGLES = 4 #DIRS = [0, 90, 180, 270] for street images

def main(indicator, folder_indicator):
    """
    ...
    """
    print(f'Running VGG for {folder_indicator} indicator')
    if True: # to be run only once!
        for FOLD in tqdm(range(0, NUM_FOLDS)):
            old_csv = f'{folder_indicator}/folds/{FOLD}.csv'
            new_csv = f'{folder_indicator}/folds/fold-{FOLD}.csv'
            for i, chunk in tqdm(enumerate(pd.read_csv(old_csv, chunksize=200))):
                chunk = chunk[chunk['fold'] != 'fold']
                if chunk.empty:
                    continue
                chunk.to_csv(new_csv, mode='a' if i > 0 else 'w', header = i == 0)

    if True: # to be run only once!
        for FOLD in tqdm(range(0, NUM_FOLDS)):

            model = load_model(glob.glob(f'{folder_indicator}/models/FOLD{FOLD}*')[0])

            csv_file = f'{folder_indicator}/folds/fold-{FOLD}.csv'
            pred_file =f'{folder_indicator}/folds/pred-{FOLD}.csv'

            for i, batch in enumerate(tqdm(pd.read_csv(csv_file, chunksize=100*NUM_ANGLES))):
                batch = batch[batch['fold'] == FOLD]
                batch['location'] = batch['filename'].apply(lambda x: re.sub('\-\d+\.jpg', '', x))
                location_value_counts = batch['location'].value_counts()
                batch['total'] = batch['location'].map(location_value_counts)
                batch = batch[batch['total'] == NUM_ANGLES]
                if len(batch) == 0:
                    continue
                X = [batch.iloc[i::NUM_ANGLES][[f'v{i}' for i in range(VGG_SHAPE[0])]] for i in range(NUM_ANGLES)]
                y = batch[indicator].iloc[0::NUM_ANGLES]
                pred = model.predict(X)
                y_pred = np.argmax(pred, axis=1) + 1
                loc = batch.iloc[0::NUM_ANGLES][['setor', indicator]]
                loc[indicator+'Pred'] = y_pred
                loc.to_csv(pred_file, index=None, header=i == 0, mode='a' if i > 0 else 'w')
                del batch

            del model      

if __name__ == "__main__":
    #python predict.py
    
    main(indicator='quintilAlfabetizacao',folder_indicator='literacy')
 
    main(indicator='quintilRenda',folder_indicator='income')
    
