from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import re


# nltk.download('wordnet')


def stemming(words):
    stemmer = SnowballStemmer("english")
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words


def lemmatize(words):
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(WordNetLemmatizer().lemmatize(word, pos="v"))
    return lemmatized_words


def lemmatize_stemming(words):
    return stemming(lemmatize(words))


def cleanup_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    return text


def preprocess(text, extra_cleanup=False):
    processed_words = []
    if extra_cleanup:
        text = cleanup_text(text)
    for token in simple_preprocess(text, min_len=3):
        if token not in STOPWORDS:
            processed_words.extend(lemmatize_stemming([token]))
    return processed_words
