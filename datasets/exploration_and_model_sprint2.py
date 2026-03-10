from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

user_movie_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating', fill_value=0)

new_user_ratings = {
    242: 5,  # Movie ID 242 rated 5
    302: 4,  # Movie ID 302 rated 4
    377: 3,  # Movie ID 377 rated 3
    346: 2,  # Movie ID 346 rated 2
    1090: 1, # Movie ID 1090 rated 1
    51: 5,   # Movie ID 51 rated 5
    225: 4,  # Movie ID 225 rated 4
    203: 3,  # Movie ID 203 rated 3
    476: 2,  # Movie ID 476 rated 2
    204: 1   # Movie ID 204 rated 1
}

new_user_vector = np.zeros(user_movie_matrix.shape[1])  
for movie_id, rating in new_user_ratings.items():
    if movie_id in user_movie_matrix.columns:
        new_user_vector[user_movie_matrix.columns.get_loc(movie_id)] = rating

user_movie_matrix_normalized = normalize(user_movie_matrix, axis=1)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_movie_matrix_normalized)

new_user_vector_normalized = normalize([new_user_vector], axis=1)

distances, indices = knn_model.kneighbors(new_user_vector_normalized, n_neighbors=3)

closest_users_ids = user_movie_matrix.index[indices[0]]

top_rated_movies = (
    user_movie_matrix.loc[closest_users_ids]
    .mean(axis=0)
    .sort_values(ascending=False)
    .head(10)