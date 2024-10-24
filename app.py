import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

# Tell NLTK to use the local nltk_data folder
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

ps = PorterStemmer()

# Load your model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Classifier")

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize
    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric tokens
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Apply stemming
    
    return " ".join(y)

input_msg = st.text_input("Enter the Email/Message")

if st.button('Predict'):
    # 1. Preprocess the input
    new_text = transform_text(input_msg)
    # 2. Vectorize the input
    vector_input = tfidf.transform([new_text])
    # 3. Make prediction
    result = model.predict(vector_input)[0]
    # 4. Display result
    if result == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')
