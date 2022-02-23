# vendor-search-project

OBJECTIVE
Normalise and predict Field categories, speciality and sub categories to provide user accessibility to an intuitive data shopping process.

DATA SOURCE

Dataset is a manually created set. Main variables considered for prediction are Label and Field description.

REQUIREMENTS

Python V3.0
nltk V3.5
sklearn V0.24.1
imblearn V0.8.0
pickle V4.0
pandas 1.1.3
numpy 1.19.2
re V2.2.1

PROCESS FLOW

## TRAINING MODEL ##
1. Field Category Model

Data : trainingdata.csv
*this is ringlead field set (as of 12-5-20) with correct field category attached. Incorporate with current database if possible. 

File : Field_Category_Model_final.py

This file is initially run to train the model with a predefined, corrected field categories. The tfidf and feature pickle files created from this model will be used to predict the categories when the Ringleadmodel_final is run.

feature.pkl- tf-idf features saved and used as variables in the RingLead file
mlp_model_ringlead.sav - model that is saved and used in the RingLead file

## This model is only run once initially UNLESS model needs reworked to include new categories that may arise in the future  ##

## ACTUAL MODEL ##
2. Ringlead Model

Data : newdata.csv

File: RingLeadModel_final.py

-This model is the actual model that will be run on ALL fields every time a new vendor is added to the database. Field category, speciality, subcategories, tags are predicted.
-Code scans for empty field categories and predicts where necessary. 
-Tags are established based on the top 100 word pairs in ALL fields, then filters out words that are not included in the top 100, leaving popular words as tags

## GENERIC TO BOTH MODELS ##
STEP 1 : Import all necessary libraries as mentioned below.

import pickle from nltk.corpus import wordnet from nltk.stem import WordNetLemmatizer import nltk from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.feature_extraction.text import TfidfTransformer import pandas as pd import re from nltk.tokenize import word_tokenize import numpy as np from nltk import ngrams

from sklearn.model_selection import RandomizedSearchCV from sklearn.neural_network import MLPClassifier from imblearn.over_sampling import SMOTE import pickle from nltk.corpus  import nltk from sklearn.feature_extraction.text import TfidfVectorizer  import re  from sklearn.model_selection import train_test_split from sklearn.metrics import confusion_matrix from sklearn import metrics from nltk.tokenize import word_tokenize from nltk.corpus import stopwords from imblearn.under_sampling import RandomUnderSampler

nltk.download(‘wordnet’) nltk.download(‘wordtokenize’) nltk.download(‘stopwords’) nltk.download(‘punkt’) nltk.download('averaged_perceptron_tagger')nltk.download('WordNetLemmatizer')nltk.download(‘ngrams’) 

STEP 2 : Input the respective dataset as mentioned above.

STEP 3 : Run the respective codes and the field categories are predicted accordingly to that particular vendor.

## APPENDIX ##
RingLeadFields.csv- Completed current fields as a result of all modeling and manual categorization. Should be applied initially to existing DB rather than running models. 

newdata.csv- Simulates new data added to the database that have no categories

updateddfields.csv- The result of running the RingLead model. Simulates completed fields with all new columns attached. 

trainingdata.csv- Current ringlead fields with just Field Category manually attached, used to train and test model, set is assumed to be 100% accurate
