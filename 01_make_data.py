# Importing libraries
import pandas as pd

# Load required CSV files
genome_scores = pd.read_csv('ml-25m/genome-scores.csv')
genome_tags = pd.read_csv('ml-25m/genome-tags.csv')
movies = pd.read_csv('ml-25m/movies.csv')
ratings = pd.read_csv('ml-25m/ratings.csv')
tags = pd.read_csv('ml-25m/tags.csv')

# Minimum number of ratings for a movie to be considered
min_ratings = 100

# Calculate the number of ratings for each movie
ratings_count = ratings.groupby('movieId').size()

# Filter movies by the minimum number of ratings
filtered_ratings_movies = ratings_count[ratings_count >= min_ratings].index

filtered_ratings = (
    ratings[ratings['movieId'].isin(filtered_ratings_movies)]
)

filtered_ratings.reset_index(drop = True, inplace = True)

# Merge movies with ratings (filtered_ratings)
movies_ratings = pd.merge(movies, filtered_ratings, on = 'movieId')

# Merge movies_ratings with tags
dataset_with_tags = pd.merge(movies_ratings, tags, on=['movieId', 'userId'], how = 'left')

# Merge genome scores with genome tags to associate tag names with their relevance scores
genome_scores_tags = pd.merge(genome_scores, genome_tags, on = 'tagId')

# Merge dataset_with_tags with genome_scores_tags to associate tag names with their relevance scores
final_dataset = pd.merge(dataset_with_tags, genome_scores_tags, on = ['movieId', 'tag'])

final_dataset.reset_index(drop = True, inplace = True)

# Save the main dataset without genome information
final_dataset.to_csv('final_dataset.csv', index = False)