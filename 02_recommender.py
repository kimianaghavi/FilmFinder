import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process

def load_data():
    """
    Load Movies data.
    """
    movies = pd.read_csv('ml-25m/movies.csv')
    return movies

def preprocess_genres(movies):
    """
    Convert genres to TF-IDF features for similarity comparison.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    # TODO: Improve this with more than genres (tags, etc.)
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'].str.replace('|', ' '))
    return tfidf_matrix, tfidf_vectorizer.get_feature_names_out()

def get_closest_matches(user_input, movies):
    """
    Find the closest matches to the user's input using fuzzy matching.
    """
    titles = movies['title'].tolist()
    closest_matches = process.extract(user_input, titles, limit=5)
    return closest_matches

def get_recommendations(movie_id, tfidf_matrix, movies):
    """
    Get movie recommendations based on genre similarity using movieId.
    """
    movie_idx = movies.index[movies['movieId'] == movie_id].tolist()[0]
    cosine_similarities = linear_kernel(tfidf_matrix[movie_idx:movie_idx+1], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[:-6:-1]
    return movies.iloc[similar_indices]['movieId'].tolist(), cosine_similarities[similar_indices]

def describe_similarity(score):
    """
    Convert similarity score to a descriptive text.
    """
    if score >= 0.9:
        return "Highly Similar"
    elif score >= 0.75:
        return "Very Similar"
    elif score >= 0.5:
        return "Similar"
    else:
        return "Somewhat Similar"

def display_recommendations(recommended_movie_ids, movies, scores):
    """
    Display recommended movies along with a user-friendly reason for recommendation.
    """
    # TODO: Update after adding "more than genres" part
    print("\nRecommended Movies based on Genre Similarity:")
    for movie_id, score in zip(recommended_movie_ids, scores):
        movie = movies[movies['movieId'] == movie_id].iloc[0]
        similarity_description = describe_similarity(score)
        print(f"{movie['title']} (Genres: {movie['genres'].replace('|', ', ')}) - {similarity_description}")


def recommend():
    """
    Main recommendation function that interacts with the user.
    """
    movies = load_data()
    tfidf_matrix, genre_names = preprocess_genres(movies)

    user_input = input("Enter a movie title to get recommendations: ")
    closest_matches = get_closest_matches(user_input, movies)

    print("\nDid you mean:")
    for i, (title, _) in enumerate(closest_matches, start=1):
        print(f"{i}. {title}")
    print("6. None of these")

    choice = int(input("Please select the correct movie (1-6): ")) - 1

    if choice == 5:
        print("Sorry, I could not find a close match to what you typed. Please try again.")
        return
    else:
        selected_title = closest_matches[choice][0]
        # Here we need to find the movieId for the selected title
        movie_id = movies[movies['title'] == selected_title]['movieId'].iloc[0]

        # Fetch recommendations using the movieId
        recommended_movie_ids, scores = get_recommendations(movie_id, tfidf_matrix, movies)

        # Now display recommendations
        display_recommendations(recommended_movie_ids, movies, scores)


if __name__ == "__main__":
    recommend()
