import re
from pathlib import Path

import pandas as pd
import spacy
import asent
from spacytextblob.spacytextblob import SpacyTextBlob


# ---------------------------------------------------------
# Pragmatic analysis: story vs non-story
# Author: Ema Slivkova
#
# This script analyzes pragmatic patterns in the development
# data. It includes:
# 1. basic sentiment scores
# 2. extra pragmatic/discourse features
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
# Word lists for pragmatic patterns
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
# Helper functions
# -----------------------------

def tokenize(text):
    """Simple tokenizer for counting word-list features."""
    return re.findall(r"\b[\w']+\b", str(text).lower())


def count_words_per_100(tokens, word_list):
    """Count how often words from a list occur per 100 words."""
    if len(tokens) == 0:
        return 0

    count = sum(1 for token in tokens if token in word_list)
    return (count / len(tokens)) * 100


def count_phrases_per_100(text, phrase_list, word_count):
    """Count phrase markers such as 'I think' or 'in my opinion' per 100 words."""
    if word_count == 0:
        return 0

    text_lower = str(text).lower()
    count = 0

    for phrase in phrase_list:
        count += text_lower.count(phrase)

    return (count / word_count) * 100


def get_compound_sentiment(text):
    """Return Asent compound sentiment for a piece of text."""
    doc = nlp(str(text))
    return doc._.polarity.compound


def get_sentiment_shift(text):
    """
    Split a text into beginning, middle, and end.
    Then calculate how much the compound sentiment changes.
    """
    doc = nlp(str(text))
    sentences = [sent.text for sent in doc.sents]

    if len(sentences) < 3:
        return 0

    third = max(1, len(sentences) // 3)

    beginning = " ".join(sentences[:third])
    middle = " ".join(sentences[third:third * 2])
    end = " ".join(sentences[third * 2:])

    scores = [
        get_compound_sentiment(beginning),
        get_compound_sentiment(middle),
        get_compound_sentiment(end)
    ]

    return max(scores) - min(scores)


def analyze_pragmatics(text):
    """
    Extract sentiment and pragmatic/discourse features from one text.
    """
    text = str(text)
    tokens = tokenize(text)
    word_count = len(tokens)

    doc = nlp(text)

    return {
        "word_count": word_count,

        # Basic sentiment scores
        "asent_neg": doc._.polarity.negative,
        "asent_neu": doc._.polarity.neutral,
        "asent_pos": doc._.polarity.positive,
        "asent_compound": doc._.polarity.compound,
        "polarity": doc._.blob.polarity,
        "subjectivity": doc._.blob.subjectivity,

        # Extra pragmatic/discourse features
        "emotion_per_100": count_words_per_100(tokens, EMOTION_WORDS),
        "temporal_per_100": count_words_per_100(tokens, TEMPORAL_MARKERS),
        "stance_per_100": count_phrases_per_100(text, STANCE_MARKERS, word_count),
        "sentiment_shift": get_sentiment_shift(text)
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