# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:28:01 2021

@author: sree
"""
#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import time
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
#import random
import spacy
#import boto3
# import zipfile
#import os
# import json
from sense2vec import Sense2Vec
#import requests
# from collections import OrderedDict
#import string
# import pke
import nltk
import numpy 
from nltk import FreqDist
#nltk.download('brown')
#nltk.download('stopwords')
#nltk.download('popular')
#from nltk.corpus import stopwords
from nltk.corpus import brown
#from nltk.tokenize import sent_tokenize
from similarity import levenshtein
#from flashtext import KeywordProcessor
#from encoding import beam_search_decoding
from mcq import tokenize_sentences
from mcq import get_keywords
from mcq import get_sentences_for_keyword
#from mcq import generate_questions_mcq
from mcq import generate_normal_questions



class QGen:
    
    def __init__(self):
        
        
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('Parth/result')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # model.eval()
        self.device = device
        self.model = model
        self.nlp = spacy.load('en_core_web_sm')

        self.s2v = Sense2Vec().from_disk('s2v_old')

        self.fdist = FreqDist(brown.words())
        self.normalized_levenshtein = levenshtein
        self.set_seed(42)
        
    def set_seed(self,seed):
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
            
            
    def predict_shortq(self, payload):
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 15)
        }

        text = inp['input_text']
        sentences = tokenize_sentences(text)
        joiner = " "
        modified_text = joiner.join(sentences)

        keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )

        
        keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
        
        for k in keyword_sentence_mapping.keys():
            text_snippet = " ".join(keyword_sentence_mapping[k][:3])
            keyword_sentence_mapping[k] = text_snippet

        final_output = {}

        if len(keyword_sentence_mapping.keys()) == 0:
            # print('ZERO')
            return final_output
        else:
            
            generated_questions = generate_normal_questions(keyword_sentence_mapping,self.device,self.tokenizer,self.model)
        
            
        final_output["statement"] = modified_text
        final_output["questions"] = generated_questions["questions"]
        
        if torch.device=='cuda':
            torch.cuda.empty_cache()

        return final_output

