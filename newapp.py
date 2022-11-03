# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:55:10 2022

@author: Kolawole Olanrewaju
"""

import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords



feature_extraction= TfidfVectorizer()

spam_mail= pickle.load(open("spam.sav","rb"))
vec_mail= pickle.load(open("vector.sav","rb"))

port_stem= PorterStemmer()



def pred_spam(content):
    stemmed_content= vec_mail.fit_transform([content]).toarray()

    prediction= spam_mail.predict(stemmed_content)
    
    if (prediction[0] == 0):
       return " This Not Spam Mail"
    else:
        return "This Spam Mail"

def main():
    
    
    #giving title 
    st.title('Spam Prediction Web App')
    
    #getting the input data from the user
    
    Message = st.text_input("Enter your Mail")
    
    
    
    #code for prediction
    spam= ''
    
    #creating a button for prediction
    if st.button("spam Result"):
        spam= pred_spam(str(Message))
        
    st.success(spam)


if __name__== "__main__":
    main()
