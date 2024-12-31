from recommendation import loadData, fixData, calculateSimilarity, recommendMovies

# load, fix, calculate, and test
movies, tags = loadData('movies.csv', 'tags.csv')
movies = fixData(movies, tags)
cosineSim = calculateSimilarity(movies)

movieTitle = input("Enter a movie title: ")
recommendation = recommendMovies(movieTitle, movies, cosineSim)

if isinstance(recommendation, list):
    print(f"Here are some movies similar to \"{movieTitle}\":")
    for indx, rec in enumerate(recommendation, start = 1):
        print(f"{indx}. {rec}")
else:
    print(recommendation)