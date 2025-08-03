Instructions to run Song Recommendation Model:
1. Create an environment based on "requirements.txt"
2. Run "Combined_df_script" notebook using the "combined_df_no_embeddings" csv file as an input. This will produce the combined_df with lyric embeddings for use in the model. The csv is too large to upload to GitHub. This file will allow the user to search recommendations by song vs song index. Store it in your working directory. - only have to do this once. Once the csv is saved on your machine there is no need to repeat this step.
3. Download "X_full.npy" file. This file contains the numpy array with all the model data.
4. Download the model's pickle file.
5. Download and run "recommend.py" in a Python reader such as Spyder. Edit the file path variables to fit your machine. Enter a song within the database (combined_df csv) and the model will produce the top 5 song recommendations for the requested song. 
