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

from main import QGen

# New streamlit changes starts

import streamlit as st

def app():
    st.title("Frequently Asked Question Generator")
    
    st.write("1. Add your text corpus in the text area.")
    
    st.write("2. Generate simple question, answer and the context.")
    st.write("3. The total question generation count is limited to 15")
    
    raw_text = st.text_area("Enter the content or corpus from the topic the questions needed to be generated")
    
    if st.button("Generate"):
        qg = QGen()  
        payload = {"input_text": raw_text}         
            
        output = qg.predict_shortq(payload) 
        df = pd.DataFrame.from_dict(output['questions'])
        df = df.drop(labels=["id"],axis=1)        
        st.write(df)
    
    st.subheader(' ------------------------FAQ Generator---------------------- :sunglasses:')

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
