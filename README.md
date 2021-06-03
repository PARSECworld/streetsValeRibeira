![alt text](http://parsecproject.org/wp-content/uploads/2019/10/cropped-PARSEC_Logo-1.png)

# A Deep-learning Method For The Prediction Of Socio-economic Indicators From Street-view Imagery, Using A Case Study From Brazil

This code was used to generate results and images for the submitted article "A Deep-learning Method For The Prediction Of Socio-economic Indicators From Street-view Imagery, Using A Case Study From Brazil". (2021) MACHICAO, Jeaneth; PIZZIGATTI CORRÊA Pedro; FERRAZ, Katia; VELLENICH, Danton Ferreira; DAVID Romain; MABILE Laurence; STALL, Shelley; SPECHT, Alison; O'BRIEN, Margaret; MENEGUZZI, Leandro; OMETTO, Jean; SANTOS, Solange

For complete reproducibility and replication, see notes in “Reproducibility and Replication” below.

## Links to related projects

This work was based on the [paper](https://www.nature.com/articles/s41598-019-42036-w) with source code in [github](https://github.com/esrasuel/measuring-inequalities-sview)

## REQUIREMENTS:

### SOFTWARE:
Main software requirements:
Python 3.7
TensorFlow 2.1 and TensorFlow 2.2
For a  complete list of libraries and packages see list in myenv.yml, with conda 4.8.3. 

Tip: Once conda is installed, to set up conda environment run: “conda env create -f myenv.yml”


### HARDWARE:
The experiments were executed on a system with minimum configuration of:
OS: Ubuntu 16.04.6 LTS
CPU: Intel Xeon Silver 4110
GPU: 1x NVIDIA Titan Xp
Memory RAM: 8 GB
Storage: 22 GB



## GUIDE:
The sequence of steps to achieve result, including expected inputs and outputs, is presented on the following diagram:
![alt text](https://github.com/PARSECworld/streetsValeRibeira/blob/main/readme_dataflow.pptx)

Therefore, the sequence of execution is:
Data set preparation
+ 1. Indicators Dataset: Export data from IBGE (Brazilian institute for geography and statistics) for the region in study
	IDHMS/IDHMS.ipynb
+ 2. Images Dataset - Export data from Google Street View for the region in study
			0. Run_crawler.sh
			1. StreetImagesCrawler.ipynb
			2. StreetImagesAnalysisPlots.ipynb

+ 3. Data Aggregation (1_DataAggregation.ipynb) - Prepare data files for use
+ 4. Image Feature (2_ImageFeatureExtraction.py) - This script uses VGG16 pre-trained network weights to extract vectors from each of the street level images used. 
+ 5. Train (3_Train.py)  - Train convolutional neural network models 
+ 6. Prediction (4_Predict.py) - Generate predictions for other images
+ 7. Plot Predictions (5_Plots.ipynb) - Plot charts with predictions and results from study


## REPRODUCIBILITY AND REPLICATION:
In order to reproduce and replicate this experiment, all data used are available, including imagery dataset on file images.zip (5GB).

All other information needed, such as seeds and flags, should be in codes. 
One specifically graph joining images and charts used on the article was manually composed using Google Slides, therefore is not part of this content.



## Unpacking

uncompress the following zips into their corresponding directories:

+ vgg_features.csv.zip
+ geo.csv.zip

+ raw_data/ibge2010.zip
+ raw_data/shapefiles.zip

+ street_crawler/output.csv.zip
+ street_crawler/output1.csv.zip
+ street_crawler/scrapy.cfg
+ street_crawler/streets_lat_long_curated.csv.zip
+ street_crawler/streets_lat_long_failed_request.csv.zip

