import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
#nltk.download('punkt')
#nltk.download('stopwords')

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation, and perform stemming
    stemmer = PorterStemmer()
    text = [stemmer.stem(i) for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join(text)


tfidf = pickle.load(open('vectroizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("INBOX Defender(spam-prevention)")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
