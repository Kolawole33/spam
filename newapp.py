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



feature_extraction= TfidfVectorizer(min_df=1,stop_words="english",lowercase=True)

spam_mail= pickle.load(open("spam.sav","rb"))

port_stem= PorterStemmer()



def pred_spam(content):
    stemmed_content = re.search("[^a-zA-Z]"," ",str(content))
    stemmed_content= stemmed_content.lower()
    stemmed_content= stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words("english")]
    stemmed_content=" ".join(stemmed_content)
    
    stemmed_content= feature_extraction.fit(content)
    
    stemmed_content= feature_extraction.transform(stemmed_content)

    prediction=spam_mail.predict(stemmed_content)
    
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
        spam= pred_spam([Message])
        
    st.success(spam)


if __name__== "__main__":
    main()