import lyricsgenius
import pandas as pd
import time
from tqdm import tqdm

# ==== SET YOUR GENIUS API TOKEN ====
GENIUS_API_TOKEN = "lWBc3eivTr7Q3g0DcSl7hjpm-eUT7LaZtwL-cBCVk3wcGb1efmqBhq8mLDiJmutl"

# ==== INIT Genius API ====
genius = lyricsgenius.Genius(
    GENIUS_API_TOKEN,
    skip_non_songs=True,
    remove_section_headers=True,
    timeout=15
)
genius.verbose = False

# ==== STEP 1: LOAD SONGS FROM CSV ====
def load_songs_from_csv(filepath):
    df = pd.read_csv(filepath)
    required_cols = {'Song', 'Artist'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    return df.to_dict(orient='records')

# ==== STEP 2: GET LYRICS ====
def enrich_with_lyrics(tracks):
    for track in tqdm(tracks, desc="Fetching lyrics"):
        try:
            song = genius.search_song(track['Song'], track['Artist'])
            if song and song.lyrics:
                raw_lyrics = song.lyrics.strip()

                # Optional cleaning
                if "Read More" in raw_lyrics:
                    cleaned = raw_lyrics.split("Read More", 1)[1].strip()
                elif "Lyrics" in raw_lyrics:
                    cleaned = raw_lyrics.split("Lyrics", 1)[1].strip()
                else:
                    cleaned = raw_lyrics

                track['lyrics'] = cleaned
            else:
                track['lyrics'] = None
        except Exception as e:
            print(f"❌ Failed for {track['Song']} by {track['Artist']}: {e}")
            track['lyrics'] = None
        time.sleep(1)  # Respect Genius rate limits
    return tracks

# ==== MAIN ====
if __name__ == "__main__":
    csv_input_path = "songs_genre_clean.csv"  
    csv_output_path = "songs_genre_lyrics.csv"  # Use a different name for the test

    print("📥 Loading songs from CSV...")
    tracks = load_songs_from_csv(csv_input_path)

    print("🎤 Fetching lyrics from Genius...")
    enriched_tracks = enrich_with_lyrics(tracks)

    print("💾 Saving output to CSV...")
    pd.DataFrame(enriched_tracks).to_csv(csv_output_path, index=False)
    print(f"✅ Done. Saved to {csv_output_path}")
