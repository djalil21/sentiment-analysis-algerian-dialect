import tensorflow as tf

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import unicodedata
from sklearn.model_selection import train_test_split
import emoji
from emoji import EMOJI_DATA
from aaransia import transliterate, SourceLanguageError
from langdetect import detect
from deep_translator import GoogleTranslator

"""# DATA"""

dz = pd.read_excel('/dz-word2vec.xlsx',names=["text"])
ar1 = pd.read_excel('/ar-word2vec-part1.xlsx',names=["text"]) 
ar2 = pd.read_excel('/ar-word2vec-part2.xlsx',names=["text"])

lst = [dz, ar1, ar2] 
df= pd.concat(lst, ignore_index=True)

df =df.sample(n=5000)

stopwords= pd.read_excel("/stopwords.xlsx")

"""# PREPROCESS"""

# search your emoji
def is_emoji(s):
    return s in EMOJI_DATA

def add_space(text):
  result = ''
  for char in text:
    if is_emoji(char):
      result += ' '
    result += char
  return result.strip()

from langdetect import detect
from deep_translator import GoogleTranslator 

#simple function to detect and translate text 
def to_arab(text):
    text_words = []
    words = text.split(" ")
    for word in words:
      try:    
        result_lang = detect(word)
        if result_lang=="fr" or result_lang=="en":
          word=GoogleTranslator(source=result_lang, target="ar").translate(text=word)  
        else:
          word= transliterate(word, source='al', target='ar', universal=True)
      except:
        pass
      text_words.append(word)

    return ' '.join(text_words)

from camel_tools.dialectid import DialectIdentifier

did = DialectIdentifier.pretrained()
def is_MSA(text):
  predictions = did.predict([text], 'region')
  return [p.top for p in predictions] != ['Maghreb']

from pyarabic.araby import is_arabicrange, strip_tashkeel
from tashaphyne.stemming import ArabicLightStemmer

def stemming(text):
    stemmer = ArabicLightStemmer()
    words = text.split()
    stemmed_words = []

    for word in words:
        if is_MSA(word):
            # Stemming is performed only on Modern Standard Arabic words
            stemmed_word = stemmer.light_stem(word)
            stemmed_words.append(stemmed_word)
        else:
            stemmed_words.append(word)

    return ' '.join(stemmed_words)

#Remove stop words 
def remove_stp_words(text):
    text_words = []
    words = text.split(" ")
    stop_words = stopwords
    for word in words:
        if word not in stop_words:
            text_words.append(word)
    return ' '.join(text_words)

def remove_consecutive_duplicates(string):
    pattern = r'(\w)\1{2,}'
    result = re.sub(pattern, r'\1', string)
    return result

# Define preprocessing util function
def basic_preprocessing(text):
  
    #to string
    text=str(text)

    # Normalize unicode encoding
    text = unicodedata.normalize('NFC', text)
    
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
   
    #Remove URLs
    text = re.sub('http://\S+|https://\S+', '',text)

    ## Convert text to lowercases
    text = text.lower()

    #remove tashkeel
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)

    #remove repetetions
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    #remove special arab letters
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)

    #keep 2 repeat
    text = remove_consecutive_duplicates(text)

    #remove stop words
    text = remove_stp_words(text)

    #add space between emojis
    text = add_space(text)

    return text

df['prepocess1']=df.text.apply(basic_preprocessing)

df['prepocess2']=df.prepocess1.apply(to_arab)

df['prepocess3']=df.prepocess2.apply(stemming)


"""# saving"""

df.preprocess1.to_csv("word2vec-preprocess1.csv", encoding='utf-8',index=False)
df.preprocess2.to_csv("word2vec-preprocess2.csv", encoding='utf-8',index=False)
df.preprocess3.to_csv("word2vec-preprocess3.csv", encoding='utf-8',index=False)