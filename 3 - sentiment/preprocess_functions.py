#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Ten skrypt zawiera funkcje do wstępnego przetwarzania danych tekstowych dla modelu sieci neuronowej
@author: ankit.jain
"""
import os
import csv
import random
import re
import numpy as np
from keras.utils import pad_sequences

def get_data(filename):
    '''
    Odczyt danych z pliku tekstowego
    '''
    with open(filename, 'r') as f:
        target = []
        text_data = []
        lines = f.readlines()
    random.shuffle(lines)
    for line in lines:
        data = line.split('\t')
        if len(data) == 2:
            target.append(int(data[0]))
            text_data.append(data[1].rstrip())
    return text_data,target


def get_processed_tokens(text):
    '''
      Pobranie listy tokenów z recenzji
    '''
    filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)     #Usuwanie znaków interpunkcyjnych
    filtered_text = filtered_text.split()
    filtered_text = [token.lower() for token in filtered_text]
    return filtered_text


def tokenize_text(data_text, min_frequency =5):
    '''
    Tokenizacja recenzji w zbiorze danych. Odfiltrowywanie rzadko występujących tokenów.
    '''
    review_tokens = [get_processed_tokens(review) for review in data_text] # Tokenizacja zdań
    token_list = [token for review in review_tokens  for token in review] #Konwersja do pojedynczej listy
    token_freq_dict = {token:token_list.count(token) for token in set(token_list)} # Obliczanie częstotliwości występowania tokenów
    most_freq_tokens = [tokens for tokens in token_freq_dict if token_freq_dict[tokens] >= min_frequency]
    idx = range(len(most_freq_tokens))
    token_idx = dict(zip(most_freq_tokens, idx))
    return token_idx,len(most_freq_tokens)

def get_max(data):
    '''
    Obliczanie maksymalnej liczby tokenów na recenzję
    '''
    tokens_per_review = [len(txt.split()) for txt in data]
    return max(tokens_per_review)

def create_sequences(data_text,token_idx,max_tokens):
    '''
    Tworzenie sekwencji odpowiednich dla wejścia GRU
    Wejście: dane recenzji, słownik z tokenami, max_tokens
    Wyjście: padded_sequences o kształcie (len(data_text), max_tokens)
    '''
    review_tokens  = [get_processed_tokens(review) for review in data_text] # Tokenizacja zdań      
    #Konwersja tokenów na ich indeksy 
    review_token_idx = map( lambda review: [token_idx[k] for k in review if k in token_idx.keys() ], review_tokens)
    padded_sequences = pad_sequences(list(review_token_idx), maxlen=max_tokens)
    return np.array(padded_sequences)

def create_csv(token_idx,filename,output_dir):
    filename= os.path.join(output_dir,filename)
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key in token_idx.keys():
            writer.writerow([key,token_idx[key]])



