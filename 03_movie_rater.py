import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
final_dataset = pd.read_csv('final_dataset.csv')

# ----- Preprocessing the Genres -----
# Split genres into list and get a sorted list of unique genres
final_dataset['genres'] = final_dataset['genres'].str.split('|')
all_genres = sorted(list(set([genre for sublist in final_dataset['genres'].tolist() for genre in sublist])))

# Encode genres into a binary matrix
def encode_genres(genres):
    encoded = np.zeros(len(all_genres), dtype=np.float32)
    for genre in genres:
        idx = all_genres.index(genre)
        encoded[idx] = 1
    return encoded

final_dataset['genres_encoded'] = final_dataset['genres'].apply(encode_genres)

# Normalize the relevance scores
scaler = MinMaxScaler()
final_dataset['relevance'] = pd.Series(scaler.fit_transform(final_dataset['relevance'].values.reshape(-1,1)).reshape(1,-1)[0])

# ----- Prepare Training and Test Data -----
# Mapping user and movie IDs to integer indices for embedding layers
user_id_mapping = {id:i for i, id in enumerate(final_dataset['userId'].unique())}
movie_id_mapping = {id:i for i, id in enumerate(final_dataset['movieId'].unique())}
tag_id_mapping = {id:i for i, id in enumerate(final_dataset['tagId'].unique())}

# Apply mappings
final_dataset['userIndex'] = final_dataset['userId'].map(user_id_mapping)
final_dataset['movieIndex'] = final_dataset['movieId'].map(movie_id_mapping)
final_dataset['tagIndex'] = final_dataset['tagId'].map(tag_id_mapping)

# Split dataset
train, test = train_test_split(final_dataset, test_size=0.2, random_state=42)

# Prepare inputs for the model
X_train = [
    train['userIndex'].values,
    train['movieIndex'].values,
    train['tagIndex'].values,
    train['relevance'].values,
    np.stack(train['genres_encoded'].values)
]
y_train = train['rating'].values

X_test = [
    test['userIndex'].values,
    test['movieIndex'].values,
    test['tagIndex'].values,
    test['relevance'].values,
    np.stack(test['genres_encoded'].values)
]
y_test = test['rating'].values
# ----- Model Building -----
num_users = len(user_id_mapping)
num_movies = len(movie_id_mapping)
num_tags = len(tag_id_mapping)
num_genres = len(all_genres)

# Inputs
user_input = Input(shape=(1,), name='user_input')
movie_input = Input(shape=(1,), name='movie_input')
tag_input = Input(shape=(1,), name='tag_input')
relevance_input = Input(shape=(1,), name='relevance_input')
genre_input = Input(shape=(num_genres,), name='genre_input')

# Embeddings and Flatten
user_embedding = Embedding(input_dim=num_users, output_dim=50, name='user_embedding')(user_input)
user_vec = Flatten(name='user_flatten')(user_embedding)
movie_embedding = Embedding(input_dim=num_movies, output_dim=50, name='movie_embedding')(movie_input)
movie_vec = Flatten(name='movie_flatten')(movie_embedding)
tag_embedding = Embedding(input_dim=num_tags, output_dim=50, name='tag_embedding')(tag_input)
tag_vec = Flatten(name='tag_flatten')(tag_embedding)


# Concatenate features
concat = Concatenate()([user_vec, movie_vec, tag_vec, relevance_input, genre_input])
concat = BatchNormalization()(concat)

# Dense layers with dropout and regularization - to avoid overfitting
fc1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(concat)
fc1 = Dropout(0.5)(fc1)
fc2 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(fc1)
fc2 = Dropout(0.5)(fc2)
output = Dense(1, activation='linear')(fc2)

# Model compilation
model = Model(inputs=[user_input, movie_input, tag_input, relevance_input, genre_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae', 'mse'])

# Training callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
]

# Model training
history = model.fit(X_train, y_train, batch_size=4096, epochs=50, validation_split=0.2, callbacks=callbacks)

# Model evaluation
model.load_weights('best_model.keras')
y_pred = model(X_test).numpy().flatten()
rounded_predictions = []
for pred in y_pred.tolist():
    final_pred = 0
    if pred >=0.25 and pred < 0.75:
        final_pred = 0.5
    elif pred < 1.25:
        final_pred = 1.0
    elif pred < 1.75:
        final_pred = 1.5
    elif pred < 2.25:
        final_pred = 2.0
    elif pred < 2.75:
        final_pred = 2.5
    elif pred < 3.25:
        final_pred = 3.0
    elif pred < 3.75:
        final_pred = 3.5
    elif pred < 4.25:
        final_pred = 4.0
    elif pred < 4.75:
        final_pred = 4.5
    else:
        final_pred = 5.0
    rounded_predictions.append(final_pred)

evaluation_results = model.evaluate(X_test, y_test)
print(f"Test Loss: {evaluation_results[0]}, Test MAE: {evaluation_results[1]}")


# # Visualization
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('train_val_loss.png')
plt.show()

# Calculate residuals
residuals = y_test - rounded_predictions

# Plot residuals distribution
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='green')
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.savefig('residual_distribution.png')
plt.show()

# Plot distribution of actual and predicted ratings
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=30, alpha=0.5, label='Actual Ratings', color='blue')
plt.hist(rounded_predictions, bins=30, alpha=0.5, label='Predicted Ratings', color='red')
plt.title('Distribution of Actual and Predicted Ratings')
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.legend(loc='upper left')
plt.savefig('distributions.png')
plt.show()

# Calculate error metrics
mae = np.mean(np.abs(residuals))
mse = np.mean(residuals**2)
rmse = np.sqrt(mse)

# Plot error metrics histogram
plt.figure(figsize=(8, 6))
plt.bar(['MAE', 'MSE', 'RMSE'], [mae, mse, rmse], color=['blue', 'green', 'red'])
plt.title('Error Metrics')
plt.ylabel('Value')
plt.savefig('errors.png')
plt.show()
