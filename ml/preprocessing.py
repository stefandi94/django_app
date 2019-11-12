import re

import spacy
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))


def convert_list_to_text(text_list):
    text = " ".join(text_list)
    return text


def convert_text_to_list(text):
    text_list = [token for token in text.split()]
    return text_list
    
    
def lower_case(text: str):
    return text.lower()


def remove_numbers(text: str):
    cleaned_text = re.sub(r'\d+', "", text)
    return cleaned_text


def remove_special_characters(text: str):
    cleaned_text = re.sub(r'\W+', " ", text).strip()
    return cleaned_text


def spacy_text_tokenizer(text):
    nlp = spacy.load('en')
    tokens = [x.text for x in nlp(text)]
    tokens = [tok.strip() for tok in tokens if tok.strip() != ""]
    return tokens


def remove_stop_words(text):
    if type(text) == str:
        text = word_tokenize(text)
    result = " ".join([token for token in text if token not in stop_words])
    print(result)
    

def lemmatize_text(text):
    tokens = word_tokenize(text)
    text = " ".join([lemmatizer.lemmatize(token) for token in tokens])
    return text


def stem_text(text):
    tokens = word_tokenize(text)
    text = " ".join([stemmer.stem(token) for token in tokens])
    return text
    
    