import musicbrainzngs
import pylast
import pandas as pd
import time
from tqdm import tqdm

# SETUP ------------------------------------------------------------------------

# 1. Configure MusicBrainz
musicbrainzngs.set_useragent("GenreTagFinder", "1.0", "jackmetzger13@gmail.com")

# 2. Configure Last.fm API
LASTFM_API_KEY = "c664435134e5705a886a39d77eb9d216"
LASTFM_API_SECRET = "0eab85625c09e1f392fd2e1a6cc99452"
lastfm_network = pylast.LastFMNetwork(api_key=LASTFM_API_KEY, api_secret=LASTFM_API_SECRET)

# FUNCTIONS ---------------------------------------------------------------------

def get_artist_mbid(song_title, artist_name):
    """Find MusicBrainz artist ID from song title + artist name."""
    try:
        result = musicbrainzngs.search_recordings(recording=song_title, artist=artist_name, limit=1)
        recording = result['recording-list'][0]
        artist = recording['artist-credit'][0]['artist']
        return artist['id']
    except Exception as e:
        print(f"[MBID] Error finding '{song_title}' by '{artist_name}': {e}")
        return None

def get_song_level_tags(song_title, artist_name):
    """Try to get track-level tags from Last.fm."""
    try:
        track = lastfm_network.get_track(artist_name, song_title)
        tags = track.get_top_tags(limit=5)
        if tags:
            return [tag.item.name for tag in tags]
    except Exception as e:
        print(f"[Track] No track tags for '{song_title}' by '{artist_name}': {e}")
    return []

def get_artist_level_tags(artist_mbid):
    """Get artist-level tags using MBID via Last.fm."""
    try:
        artist = lastfm_network.get_artist_by_mbid(artist_mbid)
        tags = artist.get_top_tags(limit=5)
        return [tag.item.name for tag in tags]
    except Exception as e:
        print(f"[Artist] Error getting artist tags: {e}")
        return []

def get_genre_tags(song_title, artist_name):
    """Main function: Try song-level tags, then fallback to artist-level."""
    print(f"\n🎵 Looking up: '{song_title}' by '{artist_name}'")
    
    # Try track-level tags first
    track_tags = get_song_level_tags(song_title, artist_name)
    if track_tags:
        print("✅ Track-level tags found.")
        return track_tags, "track"

    # If no track tags, get MBID and fetch artist-level tags
    mbid = get_artist_mbid(song_title, artist_name)
    if mbid:
        artist_tags = get_artist_level_tags(mbid)
        if artist_tags:
            print("🔁 Falling back to artist-level tags.")
            return artist_tags, "artist"
        else:
            print("⚠️ No artist tags found.")
    else:
        print("❌ No MBID found to look up artist tags.")
    
    return [], "none"


# MAIN --------------------------------------------------------------------------

def main():
    df = pd.read_csv("initial_songs_master.csv")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing songs"):
        original_song = row['Song']
        original_artist = row['Artist']
        song_clean = row['Song_Clean']
        artist_clean = row['Artist_Clean']

        tags, source = get_genre_tags(song_clean, artist_clean)
        tags_str = ", ".join(tags) if tags else ""

        results.append({
            "Song": original_song,
            "Artist": original_artist,
            "Genre Tags": tags_str,
            "Tag Source": source
        })
        time.sleep(0.25)

    results_df = pd.DataFrame(results)
    results_df.to_csv("songs_with_genre_tags_full.csv", index=False)
    print("\n✅ Done! Results saved to 'songs_with_genre_tags_full.csv'.")


if __name__ == "__main__":
    main()

