#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:41:10 2025

@author: cristiantobar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import textwrap

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def plot_top_words(model, feature_names, n_top_words=10):
  fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
  axes = axes.flatten()
  for topic_idx, topic in enumerate(model.components_):
    top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
    top_features = [feature_names[i] for i in top_features_ind]
    weights = topic[top_features_ind]

    ax = axes[topic_idx]
    ax.barh(top_features, weights, height=0.7)
    ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=20)
    for i in "top right left".split():
        ax.spines[i].set_visible(False)
    fig.suptitle('LDA', fontsize=40)

  plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
  plt.show()

def wrap(x):
  return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)

#**************MAIN******************
nltk.download('stopwords')
df = pd.read_csv('bbc_data.csv')

#Defining the stopwords
stops = set(stopwords.words('english'))
stops = stops.union({
    'said', 'would', 'could', 'told', 'also', 'one', 'two',
    'mr', 'new', 'year', 
})
stops = list(stops) # required for later version of CountVectorizer

#Vectorizing information thrugh CounVectorizer
vectorizer = CountVectorizer(stop_words=stops)
X = vectorizer.fit_transform(df['data'])

#
lda = LatentDirichletAllocation(
    n_components=10, # default: 10
    random_state=12345,
)

lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
plot_top_words(lda, feature_names);

Z = lda.transform(X)

# Pick a random document
# Check which "topics" are associated with it
# Are they related to the true label?
np.random.seed(1)
i = np.random.choice(len(df))
z = Z[i]
topics = np.arange(10) + 1

fig, ax = plt.subplots()
ax.barh(topics, z)
ax.set_yticks(topics)
ax.set_title('True label: %s' % df.iloc[i]['labels']);
print(wrap(df.iloc[i]['data']))


