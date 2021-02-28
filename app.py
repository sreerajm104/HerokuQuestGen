# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:44:41 2021

@author: sree
"""
from flask import Flask, request, jsonify, render_template,Response
import pandas as pd
import numpy as np


from main import QGen

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
    # return "Generate Questions"

@app.route('/generate',methods=['POST'])
def generate():
    '''
    For rendering results on HTML GUI
    '''

    # topic = request.form.values() #No error
    topic = [x for x in request.form.values()]
    
    # return topic #remove post from decorater
    qg = QGen()  
    payload = {"input_text": topic[0]} 
    output = qg.predict_shortq(payload)   
    
    row_headers = list(list(output.values())[1].values())[0].keys()

    return render_template('index.html', prediction_text=output)

        
@app.route('/generatefile',methods=['POST'])
def generatefromfile():
    '''
    For rendering results on HTML GUI
    '''

    file = request.files['file']
    file.save(file.filename)    
    with open(file.filename,'r') as f:    
        str_text = f.read()
    # str_text = "Sachin Ramesh Tendulkar is a former international cricketer from India and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. He is the highest run scorer of all time in International cricket."
    
    qg = QGen()  
    payload = {"input_text": str_text} 
    output = qg.predict_shortq(payload)   
    return output

if __name__ == "__main__":
    app.run(debug=True)
