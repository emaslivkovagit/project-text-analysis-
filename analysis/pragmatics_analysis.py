import re
from pathlib import Path

import pandas as pd
import spacy
import asent
from spacytextblob.spacytextblob import SpacyTextBlob


# ---------------------------------------------------------
# Pragmatic analysis: story vs non-story
# Author: Ema Slivková
#
# This script analyzes pragmatic patterns in the development
# data (dev.csv). It includes:
# 1. basic sentiment scores (positive, neutral, negative sentiment, compound sentiment, subjectivity, polarity)
# 2. extra pragmatic/discourse features (emotional language, temporal discourse markers, personal stance markers, sentiment shift)
# 3. story vs non-story averages
# 4. written rules and explanations
# ---------------------------------------------------------


# -----------------------------
# Load NLP pipeline
# -----------------------------

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
nlp.add_pipe("asent_en_v1")
nlp.add_pipe("spacytextblob")


# -----------------------------
# File paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dev data" / "dev.csv"
OUTPUT_PATH = BASE_DIR / "pragmatic_patterns.txt"
SCORES_PATH = BASE_DIR / "pragmatic_scores.csv"

df = pd.read_csv(DATA_PATH)


# -----------------------------
# Word lists to detect pragmatic patterns
# -----------------------------

EMOTION_WORDS = {
    "happy", "sad", "angry", "afraid", "scared", "nervous",
    "excited", "disappointed", "frustrated", "worried",
    "shocked", "surprised", "embarrassed", "annoyed",
    "upset", "proud", "terrified", "anxious", "confused",
    "relieved", "hurt", "mad", "glad", "depressed",
    "joy", "fear", "anger", "sadness", "surprise"
}

TEMPORAL_MARKERS = {
    "then", "after", "before", "when", "while", "later",
    "suddenly", "eventually", "finally", "once", "meanwhile",
    "afterwards", "earlier", "yesterday", "today", "tomorrow",
    "first", "next", "lastly", "eventually"
}

STANCE_MARKERS = [
    "i think",
    "i feel",
    "i believe",
    "i guess",
    "i know",
    "i remember",
    "i realized",
    "in my opinion",
    "personally",
    "honestly",
    "for me",
    "to me"
]


# -----------------------------
# Section with helper functions
# -----------------------------

def tokenize(text):
    """Split a normal sentence into a list of tokens/words that the program can count"""
    return re.findall(r"\b[\w']+\b", str(text).lower())


def count_words_per_100(tokens, word_list):
    """Count how many words from word_list appear per 100 words.
    This makes short and long texts easier to compare fairly
    """
    if len(tokens) == 0: # prevent an error (no words to count)
        return 0

    count = sum(1 for token in tokens if token in word_list) # counting emotion words or temporal markers
    return (count / len(tokens)) * 100


def count_phrases_per_100(text, phrase_list, word_count):
    """Count stance markers such as 'I think' or 'in my opinion' per 100 words 
    (so that short adn long texts can be compared fairly).
    """
    if word_count == 0: # prevent an error 
        return 0

    text_lower = str(text).lower() 
    count = 0 

    for phrase in phrase_list:
        count += text_lower.count(phrase)

    return (count / word_count) * 100


def get_compound_sentiment(text):
    """Calculate the compound sentiment score for a piece of text using Asent."""
    doc = nlp(str(text))
    return doc._.polarity.compound # return one overall sentiment score for the text


def get_sentiment_shift(text):
    """
    Split a text into beginning, middle, and end.
    Then calculate how much the compound sentiment changes.
    Basically asking: Does the text change emotionally from beginning
    to middle to end? 
    For example, stories can have emotional movement - for example, 
    a story can start badly, then become tense, and end positiviely. 
    """
    doc = nlp(str(text))
    sentences = [sent.text for sent in doc.sents] # splitting the text into sentences

    if len(sentences) < 3: # if the text has fewer than 3 sentences, the function returns 0 (it can't properly split it)
        return 0 

    third = max(1, len(sentences) // 3) # calculate how many sentence should go into each part 

    beginning = " ".join(sentences[:third])
    middle = " ".join(sentences[third:third * 2])
    end = " ".join(sentences[third * 2:])

    scores = [
        get_compound_sentiment(beginning),
        get_compound_sentiment(middle),
        get_compound_sentiment(end)
    ]

    return max(scores) - min(scores) # the difference between the highest and the lowest sentiment score


def analyze_pragmatics(text):
    """
    Extract sentiment and pragmatic/discourse features from one text.
    The central analysis function of the script.
    """
    text = str(text) 
    tokens = tokenize(text) # use tokenize function to turn the text into separate words
    word_count = len(tokens) # count the number of words in text

    doc = nlp(text)

    return { # return a dictionary:
        "word_count": word_count,

        # Basic sentiment scores from Asent:
        "asent_neg": doc._.polarity.negative, # how negative the text is
        "asent_neu": doc._.polarity.neutral, # how neutral the text is
        "asent_pos": doc._.polarity.positive, # how positive the text is
        "asent_compound": doc._.polarity.compound, # one overall sentiment score 
        "polarity": doc._.blob.polarity, # polarity: whether the text is positive or negative 
        "subjectivity": doc._.blob.subjectivity, # subjectivity: whether the text is more objective/factual or subjective/opinion-based

        # Extra pragmatic/discourse features
        "emotion_per_100": count_words_per_100(tokens, EMOTION_WORDS), # count emotion words such as happy, sad, angry, or worried, per 100 words
        "temporal_per_100": count_words_per_100(tokens, TEMPORAL_MARKERS), # count temporal markers, such as then, later, after, or finally, per 100 words
        "stance_per_100": count_phrases_per_100(text, STANCE_MARKERS, word_count), # count stance/opinion phrases, such as I think, personally, or in my opinion, per 100 words
        "sentiment_shift": get_sentiment_shift(text) # measure how much the sentiment changes between the beginning, middle, and end of the text
    }


def midpoint_threshold(story_mean, nonstory_mean):
    """Use the midpoint between story and non-story averages as a simple rule threshold."""
    return (story_mean + nonstory_mean) / 2


def direction_text(story_mean, nonstory_mean):
    """Explain whether a feature is higher or lower in stories."""
    if story_mean > nonstory_mean:
        return "higher"
    elif story_mean < nonstory_mean:
        return "lower"
    else:
        return "equal"


# -----------------------------
# Run analysis
# -----------------------------

features = df["content"].apply(analyze_pragmatics)
features_df = pd.DataFrame(list(features))

df = pd.concat([df, features_df], axis=1)


# -----------------------------
# Compare story vs non-story
# -----------------------------

all_features = [
    "word_count",
    "asent_neg",
    "asent_neu",
    "asent_pos",
    "asent_compound",
    "polarity",
    "subjectivity",
    "emotion_per_100",
    "temporal_per_100",
    "stance_per_100",
    "sentiment_shift"
]

comparison = df.groupby("label")[all_features].mean()

story = comparison.loc["story"]
nonstory = comparison.loc["non-story"]


# -----------------------------
# Save detailed scores
# -----------------------------

df.to_csv(SCORES_PATH, index=False)


# -----------------------------
# Write pragmatic_patterns.txt
# -----------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write("Pragmatic Patterns: Story vs Non-Story\n")
    file.write("======================================\n\n")

    file.write("Average pragmatic feature values:\n")
    file.write("---------------------------------\n")
    file.write(comparison.to_string())
    file.write("\n\n")

    # Basic sentiment section
    file.write("Basic sentiment analysis\n")
    file.write("------------------------\n")
    file.write("Method: Asent and spacytextblob sentiment scores.\n")
    file.write(
        "Features: negative sentiment, neutral sentiment, positive sentiment, "
        "compound sentiment, polarity, and subjectivity.\n"
    )
    file.write(
        f"Observation: Stories had an average negative sentiment score of "
        f"{story['asent_neg']:.3f}, while non-stories had an average score of "
        f"{nonstory['asent_neg']:.3f}. Stories had an average compound sentiment "
        f"score of {story['asent_compound']:.3f}, while non-stories had an average "
        f"score of {nonstory['asent_compound']:.3f}. Stories had an average "
        f"subjectivity score of {story['subjectivity']:.3f}, while non-stories had "
        f"an average score of {nonstory['subjectivity']:.3f}.\n"
    )
    file.write(
        "Rule: Basic sentiment scores can be used as supporting evidence, but not "
        "as the main rule. If one label has clearly stronger negative, positive, "
        "compound, or subjective language, this may give weak evidence for that label.\n"
    )
    file.write(
        "Explanation: Sentiment analysis is useful for detecting attitudes and opinions, "
        "but story and non-story texts can both contain emotional or subjective language. "
        "Therefore, sentiment alone is not expected to be a strong distinction.\n\n"
    )

    # Pattern 1
    story_mean = story["emotion_per_100"]
    nonstory_mean = nonstory["emotion_per_100"]
    threshold = midpoint_threshold(story_mean, nonstory_mean)
    direction = direction_text(story_mean, nonstory_mean)

    file.write("Pattern 1: Emotional language\n")
    file.write("-----------------------------\n")
    file.write("Method: Count emotion words per 100 words.\n")
    file.write(f"Story average: {story_mean:.3f}\n")
    file.write(f"Non-story average: {nonstory_mean:.3f}\n")
    file.write(
        f"Rule: If emotion_per_100 is {direction} than approximately "
        f"{threshold:.3f}, this gives evidence for story.\n"
    )
    file.write(
        "Explanation: This pattern may help because stories often describe personal "
        "or emotional experiences. However, it can fail because non-stories on Reddit "
        "can also contain strong opinions, complaints, or emotional reactions.\n\n"
    )

    # Pattern 2
    story_mean = story["temporal_per_100"]
    nonstory_mean = nonstory["temporal_per_100"]
    threshold = midpoint_threshold(story_mean, nonstory_mean)
    direction = direction_text(story_mean, nonstory_mean)

    file.write("Pattern 2: Temporal discourse markers\n")
    file.write("-------------------------------------\n")
    file.write(
        "Method: Count temporal markers per 100 words, such as then, after, when, "
        "later, suddenly, eventually, and finally.\n"
    )
    file.write(f"Story average: {story_mean:.3f}\n")
    file.write(f"Non-story average: {nonstory_mean:.3f}\n")
    file.write(
        f"Rule: If temporal_per_100 is {direction} than approximately "
        f"{threshold:.3f}, this gives evidence for story.\n"
    )
    file.write(
        "Explanation: This pattern is useful because stories often organize events "
        "in a time sequence. It can fail when non-stories explain steps, processes, "
        "or examples in chronological order.\n\n"
    )

    # Pattern 3
    story_mean = story["stance_per_100"]
    nonstory_mean = nonstory["stance_per_100"]
    threshold = midpoint_threshold(story_mean, nonstory_mean)
    direction = direction_text(story_mean, nonstory_mean)

    file.write("Pattern 3: Personal stance markers\n")
    file.write("----------------------------------\n")
    file.write(
        "Method: Count expressions per 100 words, such as I think, I feel, "
        "I believe, personally, honestly, for me, and to me.\n"
    )
    file.write(f"Story average: {story_mean:.3f}\n")
    file.write(f"Non-story average: {nonstory_mean:.3f}\n")
    file.write(
        f"Rule: If stance_per_100 is {direction} than approximately "
        f"{threshold:.3f}, this gives evidence for story.\n"
    )
    file.write(
        "Explanation: This pattern may help when stories are written as personal "
        "experiences. However, it can fail because non-stories can also include "
        "personal opinions or arguments.\n\n"
    )

    # Pattern 4
    story_mean = story["sentiment_shift"]
    nonstory_mean = nonstory["sentiment_shift"]
    threshold = midpoint_threshold(story_mean, nonstory_mean)
    direction = direction_text(story_mean, nonstory_mean)

    file.write("Pattern 4: Sentiment shift\n")
    file.write("--------------------------\n")
    file.write(
        "Method: Split each text into beginning, middle, and end. Then calculate "
        "the difference between the highest and lowest compound sentiment score.\n"
    )
    file.write(f"Story average: {story_mean:.3f}\n")
    file.write(f"Non-story average: {nonstory_mean:.3f}\n")
    file.write(
        f"Rule: If sentiment_shift is {direction} than approximately "
        f"{threshold:.3f}, this gives evidence for story.\n"
    )
    file.write(
        "Explanation: This pattern may help because stories can develop emotionally "
        "from beginning to end. It can fail when a story has a stable emotional tone "
        "or when a non-story contains a strong change in opinion.\n\n"
    )

    file.write("Conclusion\n")
    file.write("----------\n")
    file.write(
        "The basic sentiment scores were useful as a starting point, but they do not "
        "fully capture the difference between stories and non-stories. More useful "
        "pragmatic patterns come from discourse-level features: emotional language, "
        "temporal organization, personal stance, and sentiment development across "
        "the text. These features focus on how language is used to express attitudes, "
        "personal involvement, and narrative structure."
    )


print("Done.")
print(f"Created: {OUTPUT_PATH}")
print(f"Created: {SCORES_PATH}")