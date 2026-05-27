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


features = df["content"].apply(analyze_sentiment)
features_df = pd.DataFrame(list(features))

df = pd.concat([df, features_df], axis=1)

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

with open("pragmatic_patterns.txt", "w", encoding="utf-8") as file:
    file.write("Pragmatic Patterns: Sentiment Analysis\n")
    file.write("=====================================\n\n")

    file.write("Average sentiment values for story and non-story texts:\n")
    file.write(comparison.to_string())
    file.write("\n\n")

    file.write("Pattern 1: Polarity\n")
    file.write("Method: spacytextblob polarity score.\n")
    file.write("Observation: Compare whether stories and non-stories differ in positive or negative sentiment.\n\n")

    file.write("Pattern 2: Subjectivity\n")
    file.write("Method: spacytextblob subjectivity score.\n")
    file.write("Observation: Compare whether stories are more subjective or personal than non-stories.\n\n")

    file.write("Pattern 3: Compound sentiment\n")
    file.write("Method: Asent compound sentiment score.\n")
    file.write("Observation: Compare whether stories have stronger overall sentiment than non-stories.\n\n")

    file.write("Conclusion:\n")
    file.write(
        "Sentiment analysis can show pragmatic differences between stories and non-stories. "
        "However, sentiment alone may not be enough, because some non-stories also contain opinions "
        "and some stories may be written in a neutral style."
    )

df.to_csv("pragmatic_sentiment_scores.csv", index=False)

print("Done.")
print("Created pragmatic_patterns.txt and pragmatic_sentiment_scores.csv")