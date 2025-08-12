To run the model follow these instructions.

For Jupyter Notebook: You can create a csv of random or selected songs based on song index and the model will recommend a song for each original song in the csv file. There is a chunk of code at the end you can modify to fit your needs. 

REQUIREMENTS: Download the conda environment requirements. 
1. You will need the "Playlist" folder of data. This folder contains multiple CSVs of pulled playlists from Spotify.
2. With that you will be able to run the model from start to finsih. The script will save intermediate CSV data files in your working directory.
3. final_df is the df that is fed into the model. This does not enclude embeddings as that causes the CSV to be too big for import, so it is done after importing df_final.
4. data_array.npy was created in the notebook script and is used in the Spyder app deployment.


TO USE SPYDER (more interactive):
1. Make sure both final_df and data_array.npy is in your working directory.
2. Open recommend.py in Spyder.
3. Change the file paths to fit your file structure for final_df and data_array.npy
4. Run the script.
5. Enter a song from final_df to generate a recommendation (NOTE: Spelling must be exact).
6. ENJOY!
