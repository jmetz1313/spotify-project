# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 19:12:59 2025

@author: Ryan
"""

# recommend.py

import pickle
import numpy as np
import pandas as pd

# Load trained KNN model
#Set the file path (change to fit your file structure)
KNN_model_FILEPATH = r'C:\Users\Ryan\OneDrive\Documents\Final Project\Final_Script\Data\knn_weighted_model.pkl'
with open(KNN_model_FILEPATH, 'rb') as f:
    knn_model = pickle.load(f)


# Load metadata and feature matrix
#Set Filepaths (change to fit your file structre)
final_df_FILEPATH = r'C:\Users\Ryan\OneDrive\Documents\Final Project\Final_Script\Data\final_df.csv'
data_array_FILEPATH = r'C:\Users\Ryan\OneDrive\Documents\Final Project\Final_Script\Data\data_array.npy'

final_df = pd.read_csv(final_df_FILEPATH)
data_array = np.load(data_array_FILEPATH)

# 🎯 Helper: Get index from song title
def get_index_by_title(title):
    matches = final_df[final_df['Song'].str.lower() == title.lower()]
    if matches.empty:
        raise ValueError(f"❌ Song '{title}' not found. Try checking the spelling.")
    return matches.index[0]

# 🔁 Core recommendation function
def get_recommendations_by_title(title, top_n=5):
    try:
        song_index = get_index_by_title(title)
    except ValueError as e:
        print(e)
        return []

    distances, indices = knn_model.kneighbors([data_array[song_index]], n_neighbors=top_n + 1)

    results = [
        {
            'Rank': i + 1,
            'Song': final_df.iloc[idx]['Song'],
            'Artist': final_df.iloc[idx]['Artist'],
            'Distance': round(dist, 4)
        }
        for i, (idx, dist) in enumerate(zip(indices[0][1:], distances[0][1:]))  # skip self-match
    ]
    return results

# 💬 Simple CLI interface
if __name__ == "__main__":
    title_input = input("🎧 Enter a song title: ")
    recommendations = get_recommendations_by_title(title_input, top_n=5)

    if recommendations:
        print("\n🔁 Recommended Songs:")
        for rec in recommendations:
            print(f"{rec['Rank']}. {rec['Song']} by {rec['Artist']} (Distance: {rec['Distance']})")
