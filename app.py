
import streamlit as st
import pandas as pd
from recommend import recommend_movies

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #10131c;
}
h1, h2, h3 {
    color: #f5c451;
}
.movie-card {
    padding: 14px;
    border-radius: 12px;
    background-color: #202536;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Load datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Calculate average rating and rating count
rating_summary = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

movies = movies.merge(rating_summary, on="movieId", how="left")
movies["average_rating"] = movies["average_rating"].fillna(0)
movies["rating_count"] = movies["rating_count"].fillna(0)

st.title("🎬 CineMatch")
st.write("Discover your next favourite movie.")
st.divider()

# Sidebar
st.sidebar.header("Explore Movies")

all_genres = sorted({
    genre
    for item in movies["genres"].fillna("")
    for genre in item.split("|")
    if genre and genre != "(no genres listed)"
})

selected_genre = st.sidebar.selectbox(
    "Filter by genre",
    ["All Genres"] + all_genres
)

st.sidebar.caption("Recommendations are based on movie genre similarity.")

# Filter movies
filtered_movies = movies.copy()

if selected_genre != "All Genres":
    filtered_movies = filtered_movies[
        filtered_movies["genres"].str.contains(
            selected_genre, regex=False, na=False
        )
    ]

# Movie recommendation section
st.subheader("🍿 Find Similar Movies")

movie_titles = sorted(filtered_movies["title"].dropna().unique())

if not movie_titles:
    st.warning("No movies found for this genre.")
else:
    selected_movie = st.selectbox(
        "Choose a movie you like",
        movie_titles
    )

    number_of_recommendations = st.slider(
        "Number of recommendations",
        3, 10, 5
    )

    if st.button("✨ Recommend Movies", type="primary"):
        results = recommend_movies(
            selected_movie, number_of_recommendations
        )

        if results:
            st.subheader("Movies You May Like")

            for movie_title in results:
                row = movies[movies["title"] == movie_title]

                if row.empty:
                    continue

                movie = row.iloc[0]
                rating = movie["average_rating"]

                with st.container(border=True):
                    st.markdown(f"### {movie['title']}")
                    st.write(f"**Genres:** {movie['genres'].replace('|', ', ')}")
                    st.write(
                        f"⭐ Average rating: {rating:.1f}/5"
                        if rating > 0
                        else "⭐ No ratings available"
                    )
                    st.write(
                        f"👥 Number of ratings: "
                        f"{int(movie['rating_count'])}"
                    )
        else:
            st.info("No recommendations found. Try another movie.")

# Top-rated movies section
st.divider()
st.subheader("🏆 Top-Rated Movies")

minimum_ratings = st.slider(
    "Minimum number of user ratings",
    min_value=1,
    max_value=100,
    value=20
)

top_movies = movies[
    (movies["rating_count"] >= minimum_ratings)
    & (movies["average_rating"] > 0)
].sort_values(
    "average_rating", ascending=False
).head(10)

if top_movies.empty:
    st.info("No movies match this rating threshold.")
else:
    for _, movie in top_movies.iterrows():
        st.write(
            f"**{movie['title']}** — "
            f"⭐ {movie['average_rating']:.1f}/5 "
            f"({int(movie['rating_count'])} ratings)"
        )

st.divider()
st.caption(
    "CineMatch | Built with Python, Pandas, Scikit-learn and Streamlit"
)
