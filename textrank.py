import numpy as np
from nltk.cluster.util import cosine_distance

stopwords = set(open('stopwords.txt', 'r').read().split('\n'))

def similarity(sentence1, sentence2, lambdaScaling=.001):
    words = {}
    index = 0

    for word in sentence1:
        if words.get(word) is None and word not in stopwords:
            words[word] = index
            index += 1

    for word in sentence2:
        if words.get(word) is None and word not in stopwords:
            words[word] = index
            index += 1

    vector1 = [0] * index
    vector2 = [0] * index

    for word in sentence1:
        if word not in stopwords:
            vector1[words[word]] += 1

    for word in sentence2:
        if word not in stopwords:
            vector2[words[word]] += 1

    return max(1 - cosine_distance(vector1, vector2), lambdaScaling)

def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs((new_P - P).sum())
        if delta <= eps:
            return new_P
        P = new_P

def createSummaryMatrix(sentences):
    matrix = np.zeros((len(sentences), len(sentences)))

    for i, sentence1 in enumerate(sentences):
        for j, sentence2 in enumerate(sentences):
            matrix[i][j] = similarity(sentence1, sentence2)

    # normalize the matrix row-wise
    for idx in range(len(matrix)):
        matrix[idx] /= matrix[idx].sum()

    return pagerank(matrix)

def createSummary(fileName, separator='\n', summaryLength=5):
    sentences = open(fileName, 'r', encoding='utf8').read().split(separator)
    result = createSummaryMatrix(sentences)

    rankedSentences = [item[0] for item in sorted(enumerate(result), key=lambda item: -item[1])]
    rankedSentences = rankedSentences[:summaryLength]
    summary = "\n".join(list(map(lambda index: sentences[index], sorted(rankedSentences))))

    return summary

def baseline(fileName, separator='\n', segmentSeparator='\n###\n'):
    segments = open(fileName, 'r', encoding='utf8').read().split(segmentSeparator)
    segments = list(map(lambda segment: segment.split(separator), segments))
    sentences = []
    for s in segments:
        sentences.extend(s)

    result = createSummaryMatrix(sentences)

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