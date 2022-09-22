import pickle
import json
import pandas as pd
import numpy as np
import re

from keras.preprocessing.text import Tokenizer,text_to_word_sequence 
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.utils.data_utils import get_file
from keras.models import Model
from keras.layers import Input, Embedding, merge ,LSTM, Dropout, concatenate, Dense, BatchNormalization, Lambda, TimeDistributed, Dot, dot
import keras.backend as K
from keras.callbacks import ModelCheckpoint
 
from sklearn.model_selection import train_test_split
 
from zipfile import ZipFile
from os.path import expanduser, exists
 
import datetime
import time
import streamlit as st

data_dict = pickle.load('model.pkl', 'rb')

def convert_text_to_index_array(text, dictionary):
   words=text_to_word_sequence(text)
   wordIndices=[]
   for word in words:
     if word in dictionary:
       wordIndices.append(dictionary[word])
     else:
       print("'%s' not in training corpus; ignoring." %(word))
   return wordIndices   
def find_if_duplicate_questions(ques1, ques2):
   """This prints yes if the two input questions are duplicate, else prints no."""
   tokenizer = Tokenizer(num_words=100000)
   with open('dictionary.json', 'r') as dictionary_file:
     dictionary = json.load(dictionary_file)
   MAX_SEQUENCE_LENGTH = 130
   q1_word_seq = convert_text_to_index_array(ques1,dictionary)
   q1_word_seq = [q1_word_seq]
   q2_word_seq = convert_text_to_index_array(ques2,dictionary)
   q2_word_seq = [q2_word_seq]
   q1_data = pad_sequences(q1_word_seq, maxlen=MAX_SEQUENCE_LENGTH)
   q2_data = pad_sequences(q2_word_seq, maxlen=MAX_SEQUENCE_LENGTH)
   pred = data_dict.predict([q1_data,q2_data])
   print(pred)
   if(pred > 0.5):
     return "Duplicate"
   else:
     return " Not Duplicate"             
def main():
  st.title('Duplicate Question Pairs')
  q1 = st.text_input('Enter question 1')
  q2 = st.text_input('Enter question 2')
  #prediction
  duplication=''
  if st.button('Predict'):
    duplication=find_if_duplicate_questions(q1,q2)
  st.success(duplication)
 

if __name__=='__main__':
  main()


#!streamlit run app.py & npx localtunnel --port 8501