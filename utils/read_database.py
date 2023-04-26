import pandas as pd

def read_excel_database(filename):
    df = pd.read_csv(filename)
    id = df["index"]
    artist = df['artist']
    title = df['title']
    mood = df['mood']
    return id, artist, title, mood