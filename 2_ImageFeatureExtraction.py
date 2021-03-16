from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import glob
import re
import os


IMG_FOLDER = 'images/'
OUT_FILE   = 'vgg_features.csv'
BATCH_SIZE = 200
DIRS       = [0, 90, 180, 270]
VGG_SHAPE  = (4096,) #number of features out from VGG

def main():
    '''
    ...
    '''
    pd.DataFrame(
    columns=['filename', 'direction'] + [f'v{i}' for i in range(VGG_SHAPE[0])] 
        ).to_csv(OUT_FILE, index=None)


    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input,
                  outputs=base_model.get_layer('fc2').output)#fc2:fully-connected-2


    images_filenames = list(set([
        re.sub(r'\-\d+\.jpg$', '', filename)
        for filename in glob.glob(f'{IMG_FOLDER}/*')
    ]))

    NBATCHES = len(images_filenames) // BATCH_SIZE + 1  

    for i in tqdm(range(NBATCHES), total=NBATCHES): #141
        batch_img = []
        records = []
        for image_filename in  images_filenames[i * BATCH_SIZE:(i+1) * BATCH_SIZE]:
            image_filenames = [f'{image_filename}-{direction}.jpg' for direction in DIRS]#return 4 imagesnamefile
            if all([os.path.exists(f) for f in image_filenames]):
                for direction, filename in zip(DIRS, image_filenames):
                    records.append([filename, direction])
                    img = image.load_img(filename, target_size=(224, 224))#inputs to VGG are images of 224x224
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x) #<class 'numpy.ndarray'> (1, 224, 224, 3) 
                    batch_img.append(x)
            print(image_filename)

        if len(batch_img) >= 0:
            batch_img_arr = np.concatenate(batch_img, axis=0) #(800, 224, 224, 3)
            latent = model.predict(batch_img_arr) #latent space <class 'numpy.ndarray'>, (800, 4096)
            data = pd.concat([
                pd.DataFrame.from_records(records),
                pd.DataFrame(latent)
                ], axis=1)
            data.to_csv(OUT_FILE, header=None, mode='a', index=None)


if __name__ == "__main__":
    #python ImageFeatureExtraction.py
    main()            