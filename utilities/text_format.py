#!/usr/bin/env python
# coding: utf-8

import os
import re
import json
from collections import Counter


def clean_text(text):
    text = re.sub('<.*?>', '', text) # delete <sometext>
    
    text = re.sub(r"\n.+?\n\n", '', text) #delete the title  
    text = re.sub(r"\s[.,:;]\s",' ',text) # remove artefact of removing title
    
    text = re.sub(r'[\n]+?', '', text) # get rid of all \ns

    text = re.sub(r'\[\[[^]]*\]\]', '', text) # Get rid of [[ ]]
    text = re.sub(r'\[[^]]*\]', '', text)     #   Or [ ]
    text = re.sub(r'\{\{[^}]*\}\}', '', text) # for {}
    text = re.sub(r'\{[^}]*\}', '', text)
    
    #text = re.sub(r'\([^)]+?\)', '', text)
    
    text = re.sub(r'\(\([^)]*\)\)', '', text) # delete parantheses stuff
    text = re.sub(r'\([^)]*\)', '', text)
    
    text = re.sub(r"IV\.|4\."  , "dördüncü",text)
    text = re.sub(r"V\.|5\."   , "beşinci",text)
    text = re.sub(r"III\.|3\." , "üçüncü",text)
    text = re.sub(r"II\.|2\."  , "ikinci",text)
    text = re.sub(r"I\.|1\."   , "birinci",text)
    
    text = re.sub(r"²", "kare",text)
    
    text = re.sub("\d\.", "" ,text) # digit followed by a dot
    text = re.sub("\d", "", text) # all remaining digits
    
    text = re.sub("\s[^o]\.", "" ,text) # letter followed by a dot except "o", turkish 3rd person
    
    text = text.lower()
     
    text = re.sub(r"[^\w\s\.\?!%]", "", text) # replace all but these punctuations    
    text = re.sub(r"[\?!]", ".", text) #only keep .
    #text = text.translate(str.maketrans("", "", string.punctuation))
    
    # after inspection
    text = re.sub(r"[%�]", "yüzde ",text)
    
    # Old turkish (şapka)
    text = re.sub(r"[İî]","i",text)
    #text = re.sub(r"î", "i", text)
    text = re.sub(r"â", "a", text)
    text = re.sub(r"û", "u", text)
    
    # unicode  
    text = re.sub(r"[\x85\xe9\xa0]", "", text)
    escape_sequence_re = re.compile(r'\\u[0-9a-fA-F]{2,4}')
    text = re.sub(escape_sequence_re, "", text)
    #text = re.sub(u"\\u202","",text)
    #\u3000, \u2002, \u2003 ,  \u2009 \u2008 \u200a
        
    text = re.sub(' +', ' ',text) # colapse multiple spaces
    
    return text


def filter_sentence_list(sentence_list, clean_sentences=[], bound=5):
    
    for sentence in sentence_list:
        
        if sentence and len(sentence) > bound:
        
            if sentence[-1] != '.':
                sentence += "."
                
            if sentence[0] == " ":
                sentence = sentence[1:]

            clean_sentences.append(sentence)
    
    return clean_sentences


def analyze_words(sentences):
       
    word_counter = Counter()

    for sentence in sentences:
        for word in sentence.strip('.').split(' '):
            if word:
                word_counter[word] += 1  

    print("There are {} words in the dataset.".format(len(word_counter)))

    print("\nThe 10 most common words with their number of appearances:\n")
    n_most_common = word_counter.most_common(10)
    for pair in n_most_common:
        print("{}\t |\t {}".format(pair[0],pair[1]))
        
    return word_counter


def analyze_symbols(sentences,alphabet):
    
    symbol_counter = Counter()
    for sentence in sentences:
        for symbol in sentence:              
            symbol_counter[symbol] += 1 

    print("\nThere are {} characters.".format(len(symbol_counter)))

    bad_characters = [c for c in symbol_counter.keys() if c not in alphabet]
    bad_chars_str = "".join(bad_characters)

    print("There are {} bad characters.".format(len(bad_characters)))
    print("Symbol Dict: {}".format(symbol_counter.keys()))
    
    return bad_characters, bad_chars_str


def merge_sentence_lists(list_of_texts):
    
    all_sentences = []
    
    for lst in list_of_texts:
        
        for sentence in lst:
            all_sentences.append(sentence)
                
    return all_sentences


def read_txt(file_path,encoding='utf-16'):
    
    lst = []
    with open(file_path,'r',encoding=encoding) as fp:
        line = fp.readline()
        lst.append(line.strip("\n"))
        while line:

            line = fp.readline()
            lst.append(line.strip("\n"))
            
    return lst


def METUbet_formatter(sentences_original):

    # append a dot in the end
    turkish_sentences = [sentence+'.' if sentence[-1] != '.' else sentence for sentence in sentences_original] 

    # follow de-ascifier indications in the dataset
    turkish_characters = ''.join(['ç','ğ','ı','ö','ş','ü'])
    ascii_codes = ''.join(['C','G','I','O','S','U'])
  
    turkish_sentences = [sentence.translate(sentence.maketrans(ascii_codes,turkish_characters)) for sentence in turkish_sentences]

    turkish_sentences = [sentence.lower() for sentence in turkish_sentences] # Lowercase each letter

    return turkish_sentences
