# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:44:41 2021

@author: sree
"""
from flask import Flask, request, jsonify, render_template,Response, json
import pandas as pd
import numpy as np
import requests
from werkzeug.exceptions import HTTPException
import smtplib

# from main import QGen

import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import spacy
from sense2vec import Sense2Vec
import nltk
# import numpy 
from nltk import FreqDist
nltk.download('brown')
nltk.download('stopwords')
nltk.download('popular')
from nltk.corpus import brown
from similarity import levenshtein
from mcq import tokenize_sentences
from mcq import get_keywords
from mcq import get_sentences_for_keyword
from mcq import generate_normal_questions

import time

# New streamlit changes starts

import streamlit as st

def app():
    
# Class Starts
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
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                
        def predict_shortq(self, inp):
            # inp = {
            #     "input_text": payload.get("input_text"),
            #     "max_questions": payload.get("max_questions", 15)
            # }
    
            text = inp['input_text']
            sentences = tokenize_sentences(text)
            return sentences
        def modified_text_gen(self,sentences):
            
            joiner = " "
            modified_text = joiner.join(sentences)   
            return modified_text
            
        def predict_sentences(self,sentences,modified_text):
            
            keywords = get_keywords(self.nlp,modified_text,inp['max_questions'],self.s2v,self.fdist,self.normalized_levenshtein,len(sentences) )
            return keywords
            
        def final_predict(self,keywords,sentences,modified_text):
            keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
            
            for k in keyword_sentence_mapping.keys():
                text_snippet = " ".join(keyword_sentence_mapping[k][:3])
                keyword_sentence_mapping[k] = text_snippet
    
            final_output = {}
    
            if len(keyword_sentence_mapping.keys()) == 0:
                return final_output
            else:
                generated_questions = generate_normal_questions(keyword_sentence_mapping,self.device,self.tokenizer,self.model)
            final_output["statement"] = modified_text
            final_output["questions"] = generated_questions["questions"]
            
            if torch.device=='cuda':
                torch.cuda.empty_cache()
    
            return final_output               

#Class Ends    

    st.title("Frequently Asked Question Generator")
    
    st.write("1. Add your text corpus in the text area.")
    
    st.write("2. Generate simple question, answer and the context.")
    st.write("3. The total question generation count is limited to 15")
    
    raw_text = st.text_area("Enter the content or corpus from the topic the questions needed to be generated")
    
    if st.button("Generate"):
        qg = QGen()  
        payload = {"input_text": raw_text}   
        inp = {
            "input_text": payload.get("input_text"),
            "max_questions": payload.get("max_questions", 15)
            }
        
        sentences =qg.predict_shortq(inp)
        st.write("Please wait while the sentence formation is processed")
                
        modified_text = qg.modified_text_gen(sentences)
        st.success("Sentence formation and modification are completed. Please wait for mapping")
        keywords = qg.predict_sentences(sentences,modified_text) 
        
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):  
        # Update the progress bar with each iteration.
            latest_iteration.text(f'Loading {i+1}')
            bar.progress(i + 1)
            time.sleep(0.1)
        st.success("The sentence mapping is completed. Please wait for the table to be generated") 
        # if st.button("Continue"):
        output = qg.final_predict(keywords,sentences,modified_text)
        df = pd.DataFrame.from_dict(output['questions'])
        df = df.drop(labels=["id"],axis=1)        
        st.write(df)
        
    st.subheader(' -------------------------------------FAQ Generator--------------------------------- :sunglasses:')

if __name__ == "__main__":
	app()

# New streamlit changes ends

# app = Flask(__name__)


# @app.route('/')
# def home():
#     return render_template('index.html')
#     # return "Generate Questions"

# @app.route('/generate',methods=['POST'])
# def generate():
#     '''
#     For rendering results on HTML GUI
#     '''
#     try:
        
#         # topic = request.form.values() #No error
#         topic = [x for x in request.form.values()]
        
#         # return topic #remove post from decorater
#         qg = QGen()  
#         payload = {"input_text": topic[0]} 
        
        
#         output = qg.predict_shortq(payload)   
        
#         return render_template('index.html', prediction_text=output)
    
#     except requests.exceptions.RequestException as e:
#         app.logger.error(f"Error encountered in Generate: {e}")  
#         return render_template('503.html')
        
# @app.route('/generatefile',methods=['POST'])
# def generatefromfile():
#     '''
#     For rendering results on HTML GUI
#     '''
#     try: 
#         file = request.files['file']
#         file.save(file.filename)    
#         with open(file.filename,'r') as f:    
#             str_text = f.read()
#         # str_text = "Sachin Ramesh Tendulkar is a former international cricketer from India and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. He is the highest run scorer of all time in International cricket."
        
#         qg = QGen()  
#         payload = {"input_text": str_text} 
#         output = qg.predict_shortq(payload)   
        
        
        
#         email = "sreeraj.m04@gmail.com"
        
#         for i in range(len(output['questions'])):
#             html = """\
#             <html>
#             <head></head>
#             <body>
#             <p>""" + output['questions'][i]['Question']  + """<br>
#             """ + output['questions'][i]['Answer']  + """<br>
#             """ + output['questions'][i]['context']  + """<br>
#             </p>
#         </body>
#       </html>
#       """
            
#         message = html
#         server = smtplib.SMTP("smtp.gmail.com",587)
#         server.starttls()
#         server.login("code2deploy@gmail.com","Admin@123")
#         server.sendmail("code2deploy@gmail.com",email,message)
        
        
#         return output
        
#     except requests.exceptions.RequestException as e:
#         app.logger.error(f"Error encountered in GenerateFile: {e}")
#         return render_template('503.html')

# @app.after_request
# def add_header(response):
#     """
#     Add headers to both force latest IE rendering engine or Chrome Frame,
#     and also to cache the rendered page for 10 minutes.
#     """
#     response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
#     response.headers['Cache-Control'] = 'public, max-age=600'
#     return response


# @app.errorhandler(HTTPException)
# def handle_exception(e):
#     """Return JSON instead of HTML for HTTP errors."""
#     # start with the correct headers and status code from the error
#     response = e.get_response()
#     # replace the body with JSON
#     response.data = json.dumps({
#         "code": e.code,
#         "name": e.name,
#         "description": e.description,
#     })
#     response.content_type = "application/json"
#     return response
# #
# #@app.errorhandler(503)
# #def page_not_found(error):
# #    """Custom 503 page."""
# #    return render_template('503.html')

# if __name__ == "__main__":
#     app.run(debug=True)
