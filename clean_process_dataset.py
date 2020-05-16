import nltk
import numpy as np
import pandas as pd
import os, re

print("Import section done. Beginning dataset processing...")

df_raw = pd.read_csv('stackexchange_812k.csv')
print("csv file read, df's shape is", df_raw.shape)
print(df_raw.head)
print(df_raw.index)
print(df_raw.columns)
df_text = df_raw['text']
df_text = pd.DataFrame(df_text)


def clean_line(line):
    import re
    import string
    line = line.lower()
    line = re.sub(r'\d+', '', line)                #remove digit characters
    line = re.sub(r'\[$\<][^\>]*[$\>]', '', line)  #remove html and $$ tags
    line = re.sub('[^A-Za-z]+', ' ', line)         #remove punctuation
    line = re.sub('(http|www)[^\s]+', ' ', line)   #remove URLS
    line = ' '.join (line.split())                 #remove extra spaces
    return line


df_text['text'] = df_text['text'].map(lambda x: clean_line(x))          #apply clean_line function to every value in text column 
df_text['text'] = df_text['text'].map(lambda x: nltk.word_tokenize(x))  #apply word_tokenize function to every value in text column 
df_text.rename(columns={'text': 'text_processed'}, inplace=True)        #change 'text' col.name to 'text_processed'
df_raw = df_raw.join(df_text)                                           #attach 'text_processed' to df_raw
df_raw.to_csv('Submission_Cleaned_Dataset01.csv', index=True)           #save df_raw to csv file for submission