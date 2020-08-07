# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:56:36 2020

READ IN EMOTION FILE

@author: Michael
"""

import pandas as pd
import re


def clean_text(text_to_clean):
    res = text_to_clean.lower()
    res = re.sub(r"i'm", "i am", res)
    res = re.sub(r"he's", "he is", res)
    res = re.sub(r"she's", "she is", res)
    res = re.sub(r"it's", "it is", res)
    res = re.sub(r"that's", "that is", res)
    res = re.sub(r"what's", "what is", res)
    res = re.sub(r"where's", "where is", res)
    res = re.sub(r"how's", "how is", res)
    res = re.sub(r"\'ll", " will", res)
    res = re.sub(r"\'ve", " have", res)
    res = re.sub(r"\'re", " are", res)
    res = re.sub(r"\'d", " would", res)
    res = re.sub(r"\'re", " are", res)
    res = re.sub(r"won't", "will not", res)
    res = re.sub(r"can't", "cannot", res)
    res = re.sub(r"n't", " not", res)
    res = re.sub(r"n'", "ng", res)
    res = re.sub(r"'bout", "about", res)
    res = re.sub(r"'til", "until", res)
    res = re.sub(r"_comma_", "", res)
    res = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", res)
    return res


# READ EMOTION DATA IN
train_data = pd.read_csv('./datasets/train.csv')
Prompt = train_data.values[:,3]
Utterance = train_data.values[:,5]

# create question and answer data
q = []
a = []


for i in range(1,len(Prompt)):
    if (Prompt[i] == Prompt[i-1]):
        a.append('<START> '+ clean_text(Utterance[i])+ ' <END>')
        q.append(clean_text(Utterance[i-1]))