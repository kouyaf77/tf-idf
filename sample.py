import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

np.set_printoptions(precision=2)

docs = np.array(
  # 配列の要素が1つの文章
  [
    '犬　狸　狐',
    '犬　犬　狸',
    '犬　狸　狸　狸',
    '犬'
  ]
)

vectorizer = TfidfVectorizer(use_idf=True, token_pattern=u'(?u)\\b\\w+\\b')
vecs = vectorizer.fit_transform(docs)
cs = cosine_similarity(vecs.toarray(), vecs.toarray())

print(cs)