from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en import STOP_WORDS
nlp = spacy.load("en_core_web_lg")


class KeyWordExtraction:
    """
    Custom implementation of keyword extraction
    """
    def __init__(self):
        self.d = 0.85  # Damping Factor
        self.min_diff = 1e-5
        self.node_weight = None

    def set_stopwords(self, stop_words):
        ''' Set Stopwords'''
        for word in STOP_WORDS.union(set(stop_words)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        sentences = []
        for sent in doc.sents:
            words = []
            for token in sent:
                if token.pos_ in candidate_pos and token.is_stop is False:
                    words.append(token.text.lower() if lower is True else token.text)
            sentences.append(words)
        return sentences

    def get_vocab(self, sentences):
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_token_pairs(self, window_size, sentences):
        token_pairs = list()
        for sent in sentences:
            for i, word in enumerate(sent):
                for j in range(i + 1, i + window_size):
                    if j >= len(sent):
                        break
                    pair = (word, sent[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def get_matrix(self, vocab, token_pairs):
        matrix = np.zeros((len(vocab), len(vocab)), dtype=float)
        # Build Matrix
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            matrix[i][j] = 1

        # Symmetric
        # [i][j] then doing [j][i]
        matrix = self.symmetrize(matrix)

        # Normalize matrix
        norm = np.sum(matrix, axis=0)
        matrix_norm = np.divide(matrix, norm, where=norm != 0)
        return matrix_norm

    def analyze(self, text, candidate_pos=None, window_size=4, lower=False, stopwords=None):
        if stopwords is None:
            stopwords = list()
        if candidate_pos is None:
            candidate_pos = ['NOUN', 'PROPN']

        self.set_stopwords(stopwords)
        doc = nlp(text)
        sentences = self.sentence_segment(doc, candidate_pos, lower)
        vocab = self.get_vocab(sentences)
        token_pairs = self.get_token_pairs(window_size, sentences)
        matrix = self.get_matrix(vocab, token_pairs)

        # Initialization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        while True:
            pr = (1 - self.d) + self.d * np.dot(matrix, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight

    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        keywords = []
        for i, (key, value) in enumerate(node_weight.items()):
            # print(key + ' - ' + str(value))
            keywords.append(key)
            if i > number:
                break
        return keywords
