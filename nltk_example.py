import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import collections

nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('universal_tagset')

# Task1
def count_pos(document, pos):
    """Returns the number of occurrences of words with a given part of speech
    >>> count_pos('austen-emma.txt', "NOUN")
    32000
    """
    raw = nltk.corpus.gutenberg.raw(document)
    sent_tokens = nltk.sent_tokenize(raw)
    w_tokens = []
    for sent in sent_tokens:
        tokens = [w for w in nltk.word_tokenize(sent)]# if not w in stopwords]
        w_tokens.append(tokens)
    res = nltk.pos_tag_sents(w_tokens, tagset='universal')
    cnt =0
    for sent in res:
        for word in sent:
            if word[1] == pos:
                cnt += 1
    return cnt

# Task2
def get_top_stem_bigrams(document, n):
    """Returns the n most frequent bigrams of stems.
    >>> get_top_stem_bigrams('austen-emma.txt', 3)
    [(',', 'and'), ('.', "''"), (';', 'and')]
    """
    stemmer = nltk.PorterStemmer()
    sent_tokens = [word_tokenize(s) for s in sent_tokenize(nltk.corpus.gutenberg.raw(document))]
    bigrams = []
    for s in sent_tokens:
        bigrams += nltk.bigrams([stemmer.stem(w) for w in s])
    c = collections.Counter(bigrams)
    return [b for b, f in c.most_common(n)]

# Task3
def get_same_stem(document, word):
    """Returns the list of words that have the same stem as the word given, and their frequencies.
    >>> get_same_stem('austen-emma.txt', "comfort")
    [('comfort', 64), ('comfortable', 34), ('comfortably', 11), ('comforted', 3), ('comforts', 8)]
    """
    stemmer = nltk.PorterStemmer()
    ref_stem = stemmer.stem(word.lower())
    w_tokens = []
    for s in sent_tokenize(nltk.corpus.gutenberg.raw(document)):
        w_tokens += word_tokenize(s) 
    c = collections.Counter(w_tokens)
    same_stem = []
    for key, value in c.items():
        if stemmer.stem(key.lower()) == ref_stem:
            same_stem.append((key, value))
    same_stem.sort(key=lambda tup: tup[0])  # sorts in place
    return same_stem

# Task4
def most_frequent_after_pos(document, pos):
    """Returns the most frequent word after a given part of speech, and its frequency.
    >>> most_frequent_after_pos('austen-emma.txt', "NOUN")
    (',', 5958)
    """
    raw = nltk.corpus.gutenberg.raw(document)
    sent_tokens = nltk.sent_tokenize(raw)
    w_tokens = []
    for sent in sent_tokens:
        tokens = [w for w in nltk.word_tokenize(sent)]# if not w in stopwords]
        w_tokens.append(tokens)
    res = nltk.pos_tag_sents(w_tokens, tagset='universal')
    word_list = []
    for sent in res:
        if len(sent) > 1:
            for i in range(0, len(sent) - 1):
                cur_word = sent[i+1][0]
                prev_pos = sent[i][1]
                if prev_pos == pos:
                    word_list.append(cur_word)
    c = collections.Counter(word_list)
    return c.most_common(1)[0]
            

# Task5
def get_word_tfidf(text):
    """Returns the tf.idf of each of the words that appear in the given text.
    >>> get_word_tfidf("A random text!")
    [('random', 0.6822603499745392), ('text', 0.731109304312713)]
    """
    tfidf = TfidfVectorizer(input='content',stop_words='english')
    data = [nltk.corpus.gutenberg.raw(f) for f in nltk.corpus.gutenberg.fileids()]
    tfidf.fit(data)
    result = tfidf.transform([text]).toarray()
    words = tfidf.get_feature_names_out()
    word_list = []
    for word, tfidf_val in zip(words, result[0]):
        if tfidf_val > 0:
            word_list.append((word, tfidf_val))
    word_list.sort(key=lambda tup: tup[0])  # sorts in place
    return word_list

# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
