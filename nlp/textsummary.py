import spacy
from collections import Counter
from string import punctuation

nlp = spacy.load("en_core_web_lg")


class TextSummary:
    """
    Text summary generation using spacy
    """

    def __init__(self, text, limit=10):
        self.text = text
        self.limit = limit
        self.pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']

    def find_summary(self):
        keyword = []
        doc = nlp(self.text.lower())
        for token in doc:
            if token.text in nlp.Defaults.stop_words or token.text in punctuation:
                continue
            if token.pos_ in self.pos_tag:
                keyword.append(token.text)
        word_freq = Counter(keyword)
        max_freq = word_freq.most_common(1)[0][1]
        sentence_score = {}
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        for sentence in doc.sents:
            for word in sentence:
                if word.text in word_freq.keys():
                    if sentence in sentence_score.keys():
                        sentence_score[sentence.text] += word_freq[word.text]
                    else:
                        sentence_score[sentence.text] = word_freq[word.text]
        sorted_sent = [item[0].capitalize() for item in sorted(sentence_score.items(), key=lambda x: x[1], reverse=True)]
        return ' '.join(sorted_sent[:self.limit])
