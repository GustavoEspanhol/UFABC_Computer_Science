# backend/spotify_client.py
import os
from typing import Optional
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

class SpotifyClient:
    def __init__(self, client_id: Optional[str], client_secret: Optional[str]):
        if client_id and client_secret:
            auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
            self.sp = Spotify(auth_manager=auth)
        else:
            self.sp = None

    def search_artist(self, artist_name: str):
        if not self.sp:
            return None
        res = self.sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
        items = res.get("artists", {}).get("items", [])
        if not items:
            return None
        return items[0]

    def get_artist_info(self, artist_name: str):
        info = {"query": artist_name}
        artist = self.search_artist(artist_name)
        if not artist:
            info.update({
                "found": False,
                "name": artist_name,
                "genres": [],
                "popularity": None,
                "followers": None,
                "spotify_url": None
            })
            return info

        info.update({
            "found": True,
            "name": artist.get("name"),
            "genres": artist.get("genres", []),
            "popularity": artist.get("popularity"),
            "followers": artist.get("followers", {}).get("total"),
            "spotify_url": artist.get("external_urls", {}).get("spotify")
        })
        # Optionally we could fetch top tracks or audio features for deeper musical characteristics
        return info
