import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email Classifier")

def transform_text(text):
  text = text.lower() # lower case
  text = nltk.word_tokenize(text) # tokenization i.e list of words
  y = []
  for i in text:
    if i.isalnum(): # removing special characters
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

input_msg = st.text_input("Enter the Email/Message")

if st.button('Predict'):
    # 1.preprocess
    new_text = transform_text(input_msg)
    # 2.vectorize
    vector_input = tfidf.transform([new_text])
    # 3.predict
    result = model.predict(vector_input)[0]
    # 4.display
    if result == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')

