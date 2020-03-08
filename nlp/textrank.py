from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from gensim.utils import simple_preprocess
import numpy as np
from operator import itemgetter


def sentence_similarity(sent1, sent2):
    sent1 = [w.lower() for w in simple_preprocess(sent1)]
    sent2 = [w.lower() for w in simple_preprocess(sent2)]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def _pagerank(A, eps=0.0001, d=0.5):
    """
    1. eps: stop the algorithm when the difference between 2 consecutive iterations is smaller or equal to eps
    2. d: damping factor: With a probability of 1-d the user will simply pick a web page at random as the next destination, ignoring the link structure completely.
    :param A:
    :param eps:
    :param d:
    :return:
    """
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P


def build_similarity_matrix(sentences):
    S = np.zeros(len(sentences), len(sentences))
    for id1 in range(len(sentences)):
        for id2 in range(len(sentences)):
            if id1 == id2:
                continue
            S[id1][id2] = sentence_similarity(sentences[id1], sentences[id2])

        for id in range(len(S)):
            S[id] /= S[id].sum()
    return S


def textrank(sentences, top_n=5):
    S = build_similarity_matrix(sentences)
    sentence_rank = _pagerank(S)
    # Sorting sentence by rank
    sorted_sentences = [item[0] for item in sorted(enumerate(sentence_rank), key = lambda item: -1*item[1])]
    selected_sentences = sorted_sentences[:top_n]
    # itemgetter('name')({'name': 'tu', 'age': 18})  Output : "tu"
    summary = itemgetter(*selected_sentences)(sentences)
    return summary
