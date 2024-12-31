import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# load in data
def loadData(moviesPath, tagsPath):
    movies = pd.read_csv(moviesPath)
    tags = pd.read_csv(tagsPath, header = None, names = ["movieId", "tagID", "tag", "timestamp"])
    return movies, tags

# replace "|" with space
def fixData(movies, tags):
    tags = tags[pd.to_numeric(tags['movieId'], errors = 'coerce').notnull()].copy()

    movies['movieId'] = movies['movieId'].astype(int)
    tags['movieId'] = tags['movieId'].astype(int)

    # merge movies with tags based on id
    movieTags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

    # merge tags with movie data
    movies = pd.merge(movies, movieTags, left_on = 'movieId', right_on = 'movieId', how = 'left')

    # replace "|" with space
    movies.loc[:, 'genres'] = movies['genres'].str.replace('|', ' ', regex = False)

    # put genres and tags into one column
    movies['combined'] = movies['genres'] + ' ' + movies['tag'].fillna('')
    
    return movies

# calculate similarity matrix
def calculateSimilarity(movies):
    # matrix for genres
    tfidf = TfidfVectorizer(stop_words = 'english')
    tfidfMatrix = tfidf.fit_transform(movies['genres'])
    # cosine similarity
    cosineSim = cosine_similarity(tfidfMatrix, tfidfMatrix)
    return cosineSim

# recommendation

def fixTitle(title):
    return title.lower().strip()


def recommendMovies(title, movies, cosineSim):
    title = fixTitle(title)

    # reverse map of titles to indices
    movieIndices = pd.Series(movies.index, index = movies['title']).drop_duplicates()

    fixedTitles = movieIndices.index.map(fixTitle)
    
    # index of movie with same title
    if title not in fixedTitles.values:
        return f"Movie '{title}' was not found."
    
    actualTitle = movieIndices.index[fixedTitles == title][0]

    indx = movieIndices[actualTitle]

    # similarity scores
    simScores = list(enumerate(cosineSim[indx]))

    # sort by scores
    simScores = sorted(simScores, key = lambda x: x[1], reverse = True)

    # scores of 10 most similar
    simScores = simScores[1:11]

    movieIndices = [i[0] for i in simScores]

    return movies['title'].iloc[movieIndices].tolist()
