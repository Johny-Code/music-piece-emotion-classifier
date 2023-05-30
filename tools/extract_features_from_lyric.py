import os
import re
import json
import spacy
import textblob
import time
import pandas as pd
import numpy as np
from langdetect import detect


INSTRUMENTAL_COMMENT = "This song is an instrumental"
VOWELS = ['a', 'e', 'i', 'o', 'u', 'y']


def detect_language(text):
    try:
        language = detect(text)
    except Exception as e:
        print(e)
        language = None
    print(f"Detected lang = {language}")

    return language


def load_lyric_dataset(input_path, rows_to_remove):

    rows = list()
    ids = list()

    lyric_files = [os.path.join(input_path, pos_json) for pos_json in os.listdir(input_path) if pos_json.endswith('.json')]

    for file_path in lyric_files:
        with open(file_path) as f:
            song_info = json.load(f)

        try:
            id = song_info['id']
            if id in rows_to_remove:
                print(f"Song {id} is duplicated. ")
                continue
            else:
                id = id.replace("ML", "")
                id = int(id)
        except BaseException:
            id = None
            print(f"For {file_path} there is no id")

        try:
            mood = song_info['mood']
        except BaseException:
            mood = None
            print(f"For {file_path} there is no mood")

        try:
            title = song_info['title']
        except BaseException:
            title = None
            print(f"For {file_path} there is no title")

        try:
            lyric = song_info['song']['lyrics']
            if lyric == '':
                print(f"For {file_path} lyric is empty")
        except BaseException:
            lyric = None
            print(f"For {file_path} there is no lyrics")

        try:
            language = song_info['song']['language']
            if language is None:
                language = detect_language(lyric)
        except BaseException:
            print(f"For {file_path} there is no language info in dataset")
            language = detect_language(lyric)

        try:
            comment = song_info['song']['//coment']
            if comment == INSTRUMENTAL_COMMENT:
                instrumental = True
                print(f"For {file_path} is instrumental\n")
            else:
                instrumental = False
        except BaseException:
            instrumental = False

        row = (mood, title, lyric, language, instrumental)

        rows.append(row)
        ids.append(id)

    df = pd.DataFrame(rows, columns=['mood', 'title', 'lyric', 'language', 'instrumental'], index=ids)

    return df


def get_duplicated_rows(file_path):
    with open(file_path) as f:
        duplicated_info = json.load(f)

        try:
            rows_to_remove = duplicated_info['removed_rows']
        except BaseException:
            rows_to_remove = []

    return rows_to_remove


def load_en_dataset(dataset_path, duplicated_path):

    rows_to_remove = get_duplicated_rows(duplicated_path)

    dataset = load_lyric_dataset(dataset_path, rows_to_remove)

    dataset = dataset.loc[dataset['language'] == "en"]
    en_dataset = dataset.loc[dataset['instrumental'] == False]

    return en_dataset


def clean_lyric(lyric, title):

    # remove title and genius annotation
    lyric = re.sub(".+Lyrics.+\\]", '', lyric)

    # removing title (exception detected)
    lyric = re.sub(f'{title}.+Lyrics', '', lyric)

    # remove exery anotation like [Verse 1], [Chorus], [Bridge], [Part 1] etc.
    lyric = re.sub('\\[.+\\]', '', lyric)

    # remove every ********* in the lyric
    lyric = re.sub('\\*.+\\*', '', lyric)

    # remove Genius anotation "You might also like"
    lyric = re.sub('You might also like', '', lyric)

    # remove Embed exist in every lyric in the end
    if lyric[-5:] == 'Embed':
        lyric = re.sub('Embed', '', lyric)
        if lyric[-1:].isdigit():
            lyric = re.sub('\\d', '', lyric)

    # remove punctuation
    lyric = re.sub('[^\\w\\s]', '', lyric)

    # split by lines
    temp_lines = lyric.split('\n')

    # Delete empty lines
    lines = [ln for ln in temp_lines if ln != '']
    lyric = '\n '.join(lines)

    return lyric, lines


def get_word_count(tokens):
    count = 0
    for line in tokens:
        count += len(line.split(' '))
    return count


def remove_stopwords(doc, nlp):
    tks = list(filter(lambda tk: not tk.is_stop, doc))
    return spacy.tokens.Doc(nlp.vocab, words=[tk.text for tk in tks])


def get_lyrics_vector(lyric, nlp):
    doc = nlp(lyric)
    doc = remove_stopwords(doc, nlp)
    if len(doc.vector) == 300:
        return doc.vector
    else:
        print("Error in vector")
        return None


def get_echoisms(lines, nlp):
    echoism_count = 0
    for line in lines:
        doc = nlp(line)
        for i in range(len(doc) - 1):
            echoism_count += doc[i].text.lower() == doc[i + 1].text.lower()

        # count echoisms inside words e.g. yeeeeeeeah
        for token in doc:
            for j in range(len(token.text) - 1):
                if token.text[j] == token.text[j + 1] and token.text in VOWELS:
                    echoism_count += 1
                    break

    try:
        return echoism_count / get_word_count(lines)
    except ZeroDivisionError:
        return 0


def get_line_count(tokens):
    return len(tokens)


def get_duplicate_lines(lines):
    duplicate_lines = 0
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            break
        else:
            current_line = line
            try:
                next_line = lines[i + 1]
            except IndexError:
                next_line = None
            if set(current_line) == set(next_line):
                duplicate_lines += 1
    try:
        return duplicate_lines / len(lines)
    except ZeroDivisionError:
        return 0


def is_title_in_lyric(title, lines):
    for line in lines:
        if title.lower() in line.lower():
            return True
    return False


def get_verb_tense_freq(lines, nlp):
    verb_tense_freq = {'present': 0, 'past': 0, 'future': 0}

    for line in lines:
        doc = nlp(line)
        for i in range(len(doc)):
            token = doc[i]
            if token.pos_ == 'VERB' and token.tag_ != 'MD':
                if 'present' in spacy.explain(token.tag_):
                    verb_tense_freq['present'] += 1
                elif 'past' in spacy.explain(token.tag_):
                    verb_tense_freq['past'] += 1
            elif token.pos_ == 'VERB' and token.tag_ == 'MD' and (token.text.lower() == 'will' or token.text.lower() == '\'ll'):
                if i + 1 < len(doc):
                    i += 1
                    following_token = doc[i]
                    if following_token is not None and following_token.tag_ == 'VB':
                        verb_tense_freq['future'] += 1

    return verb_tense_freq['present'], verb_tense_freq['past'], verb_tense_freq['future']


def get_pos_tags_count(lines, nlp):
    pos_tags_count = {
        'ADJ': 0,  # adjective
        'ADP': 0,  # adposition
        'ADV': 0,  # adverb
        'AUX': 0,  # auxiliary
        'CCONJ': 0,  # coordinating conjunction
        'DET': 0,  # determiner
        'INTJ': 0,  # interjection
        'NOUN': 0,  # noun
        'NUM': 0,  # numeral
        'PART': 0,  # particle
        'PRON': 0,  # pronoun
        'PROPN': 0,  # proper noun
        'PUNCT': 0,  # punctuation
        'SCONJ': 0,  # subordinating conjunction
        'SYM': 0,  # symbol
        'VERB': 0,  # verb
        'X': 0,  # other
    }

    for line in lines:
        doc = nlp(line)
        for token in doc:
            if token.pos_ in pos_tags_count.keys():
                pos_tags_count[token.pos_] += 1

    return pos_tags_count


def get_sentiment(lyric):
    polarity = textblob.TextBlob(lyric).sentiment.polarity
    subjectivity = textblob.TextBlob(lyric).sentiment.subjectivity
    return polarity, subjectivity


def extract_all_features(df):

    nlp = spacy.load('en_core_web_lg')
    nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)

    rows = list()
    ids = list()
    for index, row in df.iterrows():

        title = row['title']
        lyric, lines = clean_lyric(row['lyric'], title)

        # features
        lyrics_vector = get_lyrics_vector(lyric, nlp)  # The lyrics vector created using spacy en_core_web_lg model.
        echoisms = get_echoisms(lines, nlp)                             # Percentage of echoism over the total number of words, where an echoism is either a sequence of two subsequent repeated words or the repetition of a vowel in a word.
        duplicate_lines = get_duplicate_lines(lines)
        title_in_lyric = is_title_in_lyric(title, lines)
        verb_present_freq, verb_past_freq, verb_future_freq = get_verb_tense_freq(lyric, nlp)
        pos_tags_count = get_pos_tags_count(lines, nlp)
        sentiment_polarity, sentiment_subjectivity = get_sentiment(lyric)  # TextBlob returns polarity and subjectivity of a sentence.
        # Polarity lies between [-1,1], -1 defines a negative sentiment and 1 defines a positive sentiment.
        # Subjectivity quantifies the amount of personal opinion and factual information contained in the text. lies between [0,1]
        emotion = row['mood']

        row = [emotion, lyrics_vector, echoisms, duplicate_lines, title_in_lyric, verb_present_freq, verb_past_freq, verb_future_freq,
               pos_tags_count['ADJ'], pos_tags_count['PUNCT'], sentiment_polarity, sentiment_subjectivity]

        rows.append(row)
        ids.append(index)

    features_df = pd.DataFrame(rows, columns=['emotion', 'lyrics_vector', 'echoisms', 'duplicate_lines', 'title_in_lyric',
                                              'verb_present_freq', 'verb_past_freq', 'verb_future_freq', 'count_ADJ',
                                              'count_PUNCT', 'sentiment_polarity', 'sentiment_subjectivity'], index=ids)

    return features_df


if __name__ == '__main__':
    dataset_path = os.path.join('..', 'database', 'lyrics')
    duplicate_path = os.path.join('database', 'removed_rows.json')

    en_dataset = load_en_dataset(dataset_path, duplicate_path)

    start = time.time()
    features_df = extract_all_features(en_dataset)

    end = time.time()

    feature_extraction_time = end - start
    print(f"\n\n Feature extraction took {round(feature_extraction_time, 2)} s \n\n")

    print(features_df.head())

    feature_output_path = os.path.join('..', 'database', 'features', 'lyric_features.csv')
    features_df.to_csv(feature_output_path)

    print(f"Features saved to {feature_output_path}")
