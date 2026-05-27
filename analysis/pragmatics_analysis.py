import pandas as pd
import spacy
import asent
from pathlib import Path
from spacytextblob.spacytextblob import SpacyTextBlob


# Pragmatic analysis: sentiment analysis
# Author: Ema Slivkova

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
nlp.add_pipe("asent_en_v1")
nlp.add_pipe("spacytextblob")


# Find the main project folder.
# This makes the file path work even if the script is run from another folder.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dev data" / "dev.csv"

df = pd.read_csv(DATA_PATH)


def analyze_sentiment(text):
    doc = nlp(str(text))

    return {
        "asent_neg": doc._.polarity.negative,
        "asent_neu": doc._.polarity.neutral,
        "asent_pos": doc._.polarity.positive,
        "asent_compound": doc._.polarity.compound,
        "polarity": doc._.blob.polarity,
        "subjectivity": doc._.blob.subjectivity
    }


# Run sentiment analysis for every text
features = df["content"].apply(analyze_sentiment)
features_df = pd.DataFrame(list(features))

# Add the sentiment scores to the original dataframe
df = pd.concat([df, features_df], axis=1)

# Compare average sentiment values for story and non-story texts
comparison = df.groupby("label")[
    [
        "asent_neg",
        "asent_neu",
        "asent_pos",
        "asent_compound",
        "polarity",
        "subjectivity"
    ]
].mean()

print(comparison)


# Save the pattern report inside the main project folder
OUTPUT_PATH = BASE_DIR / "pragmatic_patterns.txt"

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write("Pragmatic Patterns: Sentiment Analysis\n")
    file.write("=====================================\n\n")

    file.write("Average sentiment values for story and non-story texts:\n")
    file.write(comparison.to_string())
    file.write("\n\n")

    file.write("Pattern 1: Polarity\n")
    file.write("-------------------\n")
    file.write("Method: spacytextblob polarity score.\n")
    file.write(
        "Observation: Non-stories had a slightly higher average polarity score than stories "
        "(0.093827 vs. 0.087770). This means non-stories were a little more positive on average, "
        "but the difference was very small.\n"
    )
    file.write(
        "Interpretation: Polarity is a weak pattern because both stories and non-stories can contain "
        "positive or negative language.\n\n"
    )

    file.write("Pattern 2: Subjectivity\n")
    file.write("-----------------------\n")
    file.write("Method: spacytextblob subjectivity score.\n")
    file.write(
        "Observation: Non-stories had a slightly higher average subjectivity score than stories "
        "(0.492170 vs. 0.476215). This suggests that many non-stories in the dataset are also opinion-based.\n"
    )
    file.write(
        "Interpretation: This pattern did not support the expectation that stories would be more subjective. "
        "Reddit non-stories often include advice, opinions, and arguments, which also increases subjectivity.\n\n"
    )

    file.write("Pattern 3: Compound sentiment\n")
    file.write("-----------------------------\n")
    file.write("Method: Asent compound sentiment score.\n")
    file.write(
        "Observation: Non-stories had a slightly higher average compound sentiment score than stories "
        "(0.085086 vs. 0.069625). Stories were not clearly more emotionally intense according to this score.\n"
    )
    file.write(
        "Interpretation: Compound sentiment is also a weak pattern because some non-stories contain strong opinions, "
        "while some stories are written in a neutral style.\n\n"
    )

    file.write("Additional observation: Negative sentiment\n")
    file.write("------------------------------------------\n")
    file.write("Method: Asent negative sentiment score.\n")
    file.write(
        "Observation: Stories had a slightly higher average negative sentiment score than non-stories "
        "(0.054261 vs. 0.051910). This may reflect that some stories include problems, conflict, or difficult experiences.\n"
    )
    file.write(
        "Interpretation: The difference is very small, so this is also not a strong rule by itself.\n\n"
    )

    file.write("Conclusion:\n")
    file.write("-----------\n")
    file.write(
        "The pragmatic sentiment analysis showed only small differences between stories and non-stories. "
        "Stories had a slightly higher negative sentiment score, but non-stories had slightly higher polarity, "
        "compound sentiment, and subjectivity. This means that sentiment analysis alone is not a reliable way "
        "to distinguish stories from non-stories. However, it can still provide useful supporting evidence when "
        "combined with syntax and semantic features."
    )


# Save all sentiment scores to a CSV file
SCORES_PATH = BASE_DIR / "pragmatic_sentiment_scores.csv"
df.to_csv(SCORES_PATH, index=False)

print("Done.")
print("Created pragmatic_patterns.txt and pragmatic_sentiment_scores.csv")