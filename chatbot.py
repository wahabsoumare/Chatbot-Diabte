# from urllib import response
import numpy as np
import json
import pickle
import nltk
from nltk.stem.snowball import FrenchStemmer
from sympy import div
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

try :
    nltk.data.find('tokenizers/punkt')
    print("La ressource 'punkt' est déjà présente.")
except LookupError :
    print("La ressource 'punkt' n'a pas été trouvée. Téléchargement en cours...")
    nltk.download('punkt')
stemmer = FrenchStemmer()
intents = json.loads(open('data/intents.json', encoding = 'utf-8').read())
words = pickle.load(open('data/words.pkl', 'rb'))
classes = pickle.load(open('data/classes.pkl', 'rb'))
model = load_model('model/chatbot_model.h5')

def clean_sentence(sentence) :
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence) :
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for word in sentence_words :
        for i, w in enumerate(words) :
            if w == word :
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence) :
    bag = bag_of_words(sentence)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key = lambda x : x[1], reverse = True)
    return_list = []
    for r in results :
        return_list.append({'intent' : classes[r[0]], 'probability' : str(r[1])})
        return return_list

def get_response(intents_list, intents_json) :
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ''
    for i in list_of_intents :
        if i['tag'] == tag :
            result = np.random.choice(i['responses'])
            break
    return result

def chatbot_response(msg) :
    if not msg :
        return "Désolé, je n'ai pas compris votre message."
    intents_list = predict_class(msg)
    res = get_response(intents_list, intents)
    return res


# st.title("Chatbot Diabète")
st.markdown('<div style="text-align: center; font-size: 40px; font-weight: bold;">Chatbot Diabète</div>', unsafe_allow_html = True)
if 'messages' not in st.session_state :
    st.session_state['messages'] = []
    with st.chat_message(name = 'assistant') :
        st.markdown('Hello👋, comment puis-je vous aider?')


for message in st.session_state.messages :
    with st.chat_message(name = message['role']) :
        st.markdown(message['content'])

if prompt := st.chat_input('Comment puis-je vous aider?') :
    with st.chat_message(name = 'user') :
        st.markdown(prompt)
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })

    res = chatbot_response(prompt)
    with st.chat_message(name = 'assistant') :
        st.markdown(res)
    st.session_state.messages.append({
        'role' : 'assistant',
        'content' : res
    })