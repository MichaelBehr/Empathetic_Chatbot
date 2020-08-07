# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:45:13 2020

From Keras documentation on loading in the pre-trained Glove embedding

@author: Michael
"""

import pandas as pd
import os
import sys
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
import codecs

############### DEFINE VARIABLES

GLOVE_DIR = "./datasets/"
TEXT_DATA_DIR = "./datasets/"
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2


def get_all_conversations():
    all_conversations = []
    with codecs.open("./datasets/movie_lines.txt",
                     "rb",
                     encoding="utf-8",
                     errors="ignore") as f:
        lines = f.read().split("\n")
        for line in lines:
            all_conversations.append(line.split(" +++$+++ "))
    return all_conversations


def get_all_sorted_chats(all_conversations):
    all_chats = {}
    # get only first 10000 conversations from dataset because whole dataset will take 9.16 TiB of RAM
    for tokens in all_conversations[:10000]:
        if len(tokens) > 4:
            all_chats[int(tokens[0][1:])] = tokens[4]
    return sorted(all_chats.items(), key=lambda x: x[0])


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


def get_conversation_dict(sorted_chats):
    conv_dict = {}
    counter = 1
    conv_ids = []
    for i in range(1, len(sorted_chats) + 1):
        if i < len(sorted_chats):
            if (sorted_chats[i][0] - sorted_chats[i - 1][0]) == 1:
                if sorted_chats[i - 1][1] not in conv_ids:
                    conv_ids.append(sorted_chats[i - 1][1])
                conv_ids.append(sorted_chats[i][1])
            elif (sorted_chats[i][0] - sorted_chats[i - 1][0]) > 1:
                conv_dict[counter] = conv_ids
                conv_ids = []
            counter += 1
        else:
            continue
    return conv_dict


def get_clean_q_and_a(conversations_dictionary):
    ctx_and_target = []
    for current_conv in conversations_dictionary.values():
        if len(current_conv) % 2 != 0:
            current_conv = current_conv[:-1]
        for i in range(0, len(current_conv), 2):
            ctx_and_target.append((current_conv[i], current_conv[i + 1]))
    context, target = zip(*ctx_and_target)
    context_dirty = list(context)
    clean_questions = list()
    for i in range(len(context_dirty)):
        clean_questions.append(clean_text(context_dirty[i]))
    target_dirty = list(target)
    clean_answers = list()
    for i in range(len(target_dirty)):
        clean_answers.append('<START> '
                             + clean_text(target_dirty[i])
                             + ' <END>')
    return clean_questions, clean_answers


conversations = get_all_conversations()
total = len(conversations)
print("Total conversations in dataset: {}".format(total))
all_sorted_chats = get_all_sorted_chats(conversations)
conversation_dictionary = get_conversation_dict(all_sorted_chats)
questions, answers = get_clean_q_and_a(conversation_dictionary)
print("Questions in dataset: {}".format(len(questions)))
print("Answers in dataset: {}".format(len(answers)))


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


################ first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'ewe_uni.txt')) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))



############### LOAD AND PREPARE TEXT SAMPLES AND LABELS

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))

#################### VECTORIZE the text samples into a 2D integer tensor

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]



############### CREATING THE EMBEDDING MATRIX + EMBEDDING LAYER


# prepare embedding matrix
num_words = min(MAX_NUM_WORDS, 75000 + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)