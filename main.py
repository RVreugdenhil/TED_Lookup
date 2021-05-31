import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def load_data(path):
  dataframe = pd.read_csv(path)
  return dataframe

corpus = load_data('transcripts.csv').loc[:,'transcript']

#print(corpus.loc[:,'transcript'])
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(X.shape)
searchfor = ['cryptocurrency blockchain deep learning algorithms']
query = vectorizer.transform(searchfor)

cosine_distance = cosine_similarity(query, X)
# print(cosine_distance)







