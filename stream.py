import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

ps = PorterStemmer()

# transform_txt = pickle.load(open('transform.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_area('Enter the message')

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(ps.stem(i))


  return " ".join(y)

# transform sms
transformed_txt = transform_text(input_sms)
# vectorized sms
vectorized_txt = tfidf.transform([transformed_txt])
# predict
res = model.predict(vectorized_txt)
# display
if st.button('Predict'):
    if res == 1:
        st.header('Spam')
    else:
        st.header('Ham') 