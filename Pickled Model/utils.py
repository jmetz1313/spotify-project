# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 19:12:42 2025

@author: Ryan
"""

# utils.py

import numpy as np

def get_embedding_twitter(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def str_to_embedding(s):
    s = s.strip('[]')
    return np.array([float(x) for x in s.split()])

def get_index_by_title(title):
    matches = combined_df[combined_df['Song'].str.lower() == title.lower()]
    if matches.empty:
        raise ValueError("❌ Song not found. Double-check the title.")
    return matches.index[0]

def avg_cosine_similarity(X, k):
    """
    Computes average cosine similarity to k neighbors for each point in the dataset. Gives a measure of how clustered song embeddings are.
    
    Parameters:
        X (np.ndarray): Matrix of latent song vectors (n_songs x n_features)
        k (int): Number of neighbors
    
    Returns:
        float: Average cosine similarity across all points
    """
    # Fit Nearest Neighbors using cosine distance
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
    distances, indices = nn.kneighbors(X)

    # Remove self (first neighbor is the point itself)
    neighbor_indices = indices[:, 1:]

    similarities = []
    for i, neighbors in enumerate(neighbor_indices):
        vec = X[i]
        neighbor_vecs = X[neighbors]
        sims = cosine_similarity([vec], neighbor_vecs)[0]
        similarities.append(np.mean(sims))

    return np.mean(similarities)

#Computes the average distance to the k nearest neighbors (evaluates how spread out or clustered the data is in feature space)
def avg_neighbor_distance(X, k=5, metric='euclidean'):
    nn = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Ignore the 0 distance to self
    avg_distances = np.mean(distances[:, 1:], axis=1)

    return np.mean(avg_distances)

def collect_recommendation_frequencies(knn_model, X_test, top_n=5):
    """
    Collects and counts top-N recommendations for all test songs.
    
    Returns:
        recommendation_counter: Counter of song indices from the training set
    """
    recommendation_counter = Counter()
    
    for i in range(len(X_test)):
        test_vector = X_test[i].reshape(1, -1)
        distances, indices = knn_model.kneighbors(test_vector)
        
        top_recs = indices[0][:top_n]
        recommendation_counter.update(top_recs)
    
    return recommendation_counter

def clean_lyrics(text):
    if pd.isnull(text):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove [annotations like chorus/verse]
    text = re.sub(r"\[.*?\]", "", text)

    # Remove all punctuation (including apostrophes) - modification from week 4 to 5.
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize with spaCy
    doc = nlp(text)

    # Lemmatize and remove stopwords
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.lemma_ not in stop_words]

    return " ".join(tokens)

#Circular encoding for Camelot

def camelot_to_position(code):
    number = int(code[:-1])   # extract number (1-12)
    mode = code[-1]           # extract mode (A or B)
    mode_offset = 0 if mode == 'A' else 12        # Start at 0 if mode is A, if not start at 13
    return (number - 1) + mode_offset 

def add_sin_cos(df):
    df['camelot_pos'] = df['Camelot'].map(camelot_to_position) #map camelot positions to 0-23
    df['camelot_sin'] = np.sin(2 * np.pi * df['camelot_pos'] / 24)        #compute sine transformation 
    df['camelot_cos'] = np.cos(2 * np.pi * df['camelot_pos'] / 24)        #compute cosine transformation
    df.drop(['Camelot', 'camelot_pos'], axis=1, inplace=True)             #Drop camelot and intermediate position column.
    return df

# Helper function to transform and reattach
def encode_and_concat(df, encoder, categorical_cols):
    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    return pd.concat([df.drop(one_hot_cols, axis=1), encoded_df], axis=1)

def simple_tokenize(text):
  return text.split()

def get_fasttext_embedding(tokens, model, dim=300):
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)


