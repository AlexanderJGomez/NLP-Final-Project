import numpy as np
import string, re
from nltk.cluster.util import cosine_distance
#Just so we can get idf easily
from sklearn.feature_extraction.text import TfidfVectorizer
punctuationToRemove = string.punctuation.replace("'", "")
regex = re.compile('[%s]' % re.escape(punctuationToRemove))

stopwords = set(open('stopwords.txt', 'r').read().split('\n'))

def similarity(sentence1, sentence2, sameSegment, lambdaScaling=.001, idfs=None):
    words = {}
    index = 0
    s1 = regex.sub('', sentence1.lower()).split()
    s2 = regex.sub('', sentence2.lower()).split()
    for word in s1:
        if words.get(word) is None and word not in stopwords and (idfs is None or word in idfs):
            words[word] = index
            index += 1

    for word in s2:
        if words.get(word) is None and word not in stopwords and (idfs is None or word in idfs):
            words[word] = index
            index += 1

    vector1 = [0] * index
    vector2 = [0] * index

    for word in s1:
        if word not in stopwords and (idfs is None or word in idfs):
            vector1[words[word]] += 1

    for word in s2:
        if word not in stopwords  and (idfs is None or word in idfs):
            vector2[words[word]] += 1

    if idfs is not None:
        for word in words:
            if word not in stopwords:
                vector1[words[word]] *= idfs[word]
                vector2[words[word]] *= idfs[word]

    distance = 1 - cosine_distance(vector1, vector2)
    if sameSegment:
        distance *= 1.1
    return min(1, max(distance, lambdaScaling))


def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= eps:
            return new_P
        P = new_P

def modifiedPagerank(A, eps=0.0001, d=0.85, alpha=0.1):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= eps:
            return new_P
        P = new_P

def createSummaryMatrix(sentences, sameSegments={}, idfs=None):
    matrix = np.zeros((len(sentences), len(sentences)))

    for i, sentence1 in enumerate(sentences):
        for j, sentence2 in enumerate(sentences):
            matrix[i][j] = similarity(sentence1, sentence2, sameSegments.get((sentence1, sentence2), False), idfs=idfs)

    # normalize the matrix row-wise
    for idx in range(len(matrix)):
        matrix[idx] /= matrix[idx].sum()

    return modifiedPagerank(matrix)

def createSummary(fileName, separator='\n', summaryLength=5):
    sentences = open(fileName, 'r', encoding='utf8').read().split(separator)
    result = createSummaryMatrix(sentences)

    rankedSentences = [item[0] for item in sorted(enumerate(result), key=lambda item: -item[1])]
    rankedSentences = rankedSentences[:summaryLength]
    summary = "\n".join(list(map(lambda index: sentences[index], sorted(rankedSentences))))

    return summary


#types = "baseline", "alg1", or "alg2"
def textrank(fileName, separator='\n', paragraphSize=5, type="baseline"):
    sentences = open(fileName, 'r', encoding='utf8').read().split(separator)
    segments = [sentences[i:i + paragraphSize] for i in range(0, len(sentences), paragraphSize)]
    idfs = None
    if type == "alg2":
        vec = TfidfVectorizer(smooth_idf=False, stop_words=stopwords)
        vec.fit_transform(list(map(lambda x: "  ".join(x), segments)))
        idf_dict = vec.vocabulary_
        idfs = {k: vec.idf_[v] for k, v in idf_dict.items()}

    sameSegments = {}
    if type == "alg1":
        for s in segments:
            for sentence1 in s:
                for sentence2 in s:
                    sameSegments[(sentence1, sentence2)] = True

    result = createSummaryMatrix(sentences, sameSegments=sameSegments, idfs=idfs)

    summaries = []
    idx = 0
    for segment in segments:
        nextIdx = idx + len(segment)
        submatrix = result[idx:nextIdx]
        rankedSentences = [item[0] for item in sorted(enumerate(submatrix), key=lambda item: -item[1])]
        rankedSentences = rankedSentences[:1]
        summary = "\n".join(list(map(lambda index: sentences[idx + index], sorted(rankedSentences))))
        summaries.append(summary)
        idx = nextIdx

    return summaries

