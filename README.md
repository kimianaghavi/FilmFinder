# FilmFinder
## Overview
This project aims to build a movie recommendation system using various techniques, including content-based filtering and collaborative filtering. The system is designed to provide movie recommendations based on user input, genre similarity, and historical user ratings.

## Files
1. ### **01_make_data.py:** 
This script preprocesses the raw movie data and generates a final dataset with relevant information. It filters movies based on a minimum number of ratings and merges them with tags and genome scores to enrich the dataset. The resulting dataset is saved as final_dataset.csv.

2. ### **02_recommender.py:** 
This script implements a content-based recommendation algorithm. It uses TF-IDF vectors of movie genres to find similar movies based on user input. Fuzzy matching is used to handle potential user input errors. The script interactively asks the user for a movie title and provides recommendations based on genre similarity.

3. ### ***03_movie_rater.py:*** 
This script builds a collaborative filtering recommendation model using neural networks. It preprocesses the final dataset, normalizes relevance scores, and prepares training and test data. The model architecture includes embedding layers for user, movie, and tag IDs, as well as genre information. It uses a combination of dense layers with dropout and regularization to prevent overfitting. The script also evaluates the model's performance using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE), and visualizes the training process and error metrics.

## Usage
To run the project, follow these steps:

1. Make sure you have the necessary dependencies installed. You can install them using pip:

## Copy code
``` pip install pandas scikit-learn keras matplotlib seaborn fuzzywuzzy ```

2. Execute each script in the following order:

- 01_make_data.py
- 02_recommender.py
- 03_movie_rater.py

3. Follow the instructions provided by each script to interact with the recommendation system.

## Dependencies
- pandas
- scikit-learn
- keras
- matplotlib
- seaborn
- fuzzywuzzy

## Authors
- Kimia Naghavi
- Ali Nikan
