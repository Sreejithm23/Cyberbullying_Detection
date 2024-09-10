import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


nltk.download('stopwords')

# Load the trained model and vectorizer from the project directory
nb=pickle.load(open('nb.pkl','rb'))
vec=pickle.load(open('vec.pkl','rb'))


st.set_page_config(page_title="Project")


# Streamlit app
st.title('Cyberbullying Detection App')
st.write('Enter the text you want to classify as not-cyberbullying, race, gender, or religion:')

# Text input
user_input = st.text_area('Enter text here')


# Preprocessing functions
def preprocess_text(text):
    tk = nltk.tokenize.RegexpTokenizer(r'\w+')
    stemer = SnowballStemmer('english')
    words = set(stopwords.words('english'))

    command = pd.Series([text])
    command = command.apply(lambda x: tk.tokenize(x)).apply(lambda x: ' '.join(x))
    command = command.str.replace('[^a-zA-Z0-9]', ' ', regex=True)
    command = command.apply(lambda x: [stemer.stem(i.lower()) for i in tk.tokenize(x)]).apply(lambda x: ' '.join(x))
    command = command.apply(lambda x: [i for i in tk.tokenize(x) if i not in words]).apply(lambda x: ' '.join(x))

    return command


# Predict button
if st.button('Predict'):
    if user_input.strip() == '':
        st.write('Please enter some text.')
    else:
        # Preprocess the input text
        preprocessed_text = preprocess_text(user_input)
        input_data = vec.transform(preprocessed_text)

        # Make prediction
        prediction = nb.predict(input_data)


        # Display the result
        st.write(f'The entered text is classified as **{prediction}**.')


