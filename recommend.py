
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie information
movies = pd.read_csv("data/movies.csv")

# Prepare movie genres
movies["genres"] = movies["genres"].fillna("").str.replace(
    "|", " ", regex=False
)

# Convert genres into numerical features
vectorizer = TfidfVectorizer()
movie_vectors = vectorizer.fit_transform(movies["genres"])

# Calculate similarity between movies
similarity = cosine_similarity(movie_vectors)


def recommend_movies(movie_title, count=5):
    # Find the selected movie
    matches = movies[
        movies["title"].str.contains(
            movie_title, case=False, regex=False, na=False
        )
    ]

    if matches.empty:
        return []

    movie_index = matches.index[0]

    # Rank movies by similarity
    scores = list(enumerate(similarity[movie_index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = []

    for index, score in scores:
        if index != movie_index:
            recommendations.append(movies.iloc[index]["title"])

        if len(recommendations) == count:
            break

    return recommendations


# Test the recommendation system
if __name__ == "__main__":
    title = input("Enter a movie name: ")

    results = recommend_movies(title)

    print("\nRecommended movies:")

    if results:
        for number, movie in enumerate(results, start=1):
            print(f"{number}. {movie}")
    else:
        print("No matching movie found.")
