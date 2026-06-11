import sys
import re
import pandas as pd
import spacy
import asent
from spacytextblob.spacytextblob import SpacyTextBlob


# ---------------------------------------------------------
# Pragmatic analysis: story vs non-story
#
# How to run from the main project folder:
# python3 analysis/pragmatics_analysis.py dev_data/dev.csv dev_data/test.csv
#
# This program creates:
# pragmatic_patterns.txt
# ---------------------------------------------------------


# Load spaCy and sentiment tools
nlp = spacy.blank("en") # create a simple English spaCy pipeline
nlp.add_pipe("sentencizer") # add sentence splitting
nlp.add_pipe("asent_en_v1") # asent sentiment analysis
nlp.add_pipe("spacytextblob") # textblob sentiment analysis


# Emotion words that the program searches for
EMOTION_WORDS = {
    "happy", "sad", "angry", "afraid", "scared", "nervous",
    "excited", "disappointed", "frustrated", "worried",
    "shocked", "surprised", "embarrassed", "annoyed",
    "upset", "proud", "terrified", "anxious", "confused",
    "relieved", "hurt", "mad", "glad", "depressed",
    "joy", "fear", "anger", "sadness", "surprise"
}


# Temporal markers that the program searches for 
# useful because stories often describe events in order
TEMPORAL_MARKERS = {
    "then", "after", "before", "when", "while", "later",
    "suddenly", "eventually", "finally", "once", "meanwhile",
    "afterwards", "earlier", "yesterday", "today", "tomorrow",
    "first", "next", "lastly"
}


def tokenize(text):
    """Split text into lowercase separate words."""
    return re.findall(r"\b[\w']+\b", str(text).lower())
# lowercase to make the counting easier


def count_words_per_100(tokens, word_list):
    """Count words from a list per 100 words."""
    if len(tokens) == 0:
        return 0

    count = 0

    for token in tokens:
        if token in word_list:
            count += 1

    return count / len(tokens) * 100 # per 100 words makes short and long texts easier to compare fairly


def get_sentiment_shift(text):
    """
    Split the text into beginning, middle, and end.
    Then calculate how much the sentiment changes.
    How much does the emotional tone of text change
    from beggining to middle to end?
    """
    doc = nlp(str(text)) # text is one Reddit post from the content column
    sentences = [sent.text for sent in doc.sents] # split the text into a list of sentences

    if len(sentences) < 3:
        return 0 # if the text has fewer than 3 sentences, do not calculate sentiment shift

    third = len(sentences) // 3 # calculate how many sentence go into each part

    beginning = " ".join(sentences[:third])
    middle = " ".join(sentences[third:third * 2])
    end = " ".join(sentences[third * 2:])

    beginning_score = nlp(beginning)._.polarity.compound # get the overall sentiment score from the beginning
    middle_score = nlp(middle)._.polarity.compound
    end_score = nlp(end)._.polarity.compound

    scores = [beginning_score, middle_score, end_score] 

    return max(scores) - min(scores) # difference between the highest and lowest sentiment score
    # how strong is the sentiment shift (high sentiment shift = emotional tone changes a lot)


def analyze_text(text):
    """
    Takes one text/post and calculates all pragmatic 
    measurements for that text
    """
    tokens = tokenize(text) # separate lowercase words
    doc = nlp(str(text)) # send full text through the NLP pipeline

    return pd.Series({ # return a small table row
        # Asent sentiment scores
        "negative_sentiment": doc._.polarity.negative, # measure how negative the text is
        "neutral_sentiment": doc._.polarity.neutral,
        "positive_sentiment": doc._.polarity.positive,
        "compound_sentiment": doc._.polarity.compound, # overall sentiment score (summary)

        # spacytextblob sentiment scores
        "polarity": doc._.blob.polarity, # whether the text is positive or negative, like the compound sentiment
        "subjectivity": doc._.blob.subjectivity, # how opinion-based or personal the text is

        # Extra pragmatic features
        "emotion_per_100": count_words_per_100(tokens, EMOTION_WORDS), # counts how many emotion words appear per 100 words
        "temporal_per_100": count_words_per_100(tokens, TEMPORAL_MARKERS),
        "sentiment_shift": get_sentiment_shift(text)
    })


def accuracy(predictions, gold_labels):
    """Calculate accuracy."""
    correct = 0

    for prediction, gold in zip(predictions, gold_labels): # go through predictions and real labels together
        if prediction == gold:
            correct += 1

    return correct / len(gold_labels)


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

if len(sys.argv) != 3:
    print("Usage: python3 analysis/pragmatics_analysis.py dev_data/dev.csv dev_data/test.csv")
    sys.exit()

dev_file = sys.argv[1]
test_file = sys.argv[2]

# read both .csv files as labels:
dev_df = pd.read_csv(dev_file) # read the development file as a table
test_df = pd.read_csv(test_file)


# Analyze development data
dev_features = dev_df["content"].apply(analyze_text) # take every text in dev.csv and run analyze_text on it
dev_df = pd.concat([dev_df, dev_features], axis=1) # add the new feature scores to the table


# Analyze test data
test_features = test_df["content"].apply(analyze_text)
test_df = pd.concat([test_df, test_features], axis=1)


# These are the columns I want to calculate averages for: 
features = [
    "negative_sentiment",
    "neutral_sentiment",
    "positive_sentiment",
    "compound_sentiment",
    "polarity",
    "subjectivity",
    "emotion_per_100",
    "temporal_per_100",
    "sentiment_shift"
]

# Create a table with the average feature values for stories and non-stories:
averages = dev_df.groupby("label")[features].mean() 

# Separate the story row and the non-story row: 
story = averages.loc["story"]
nonstory = averages.loc["non-story"]


# ---------------------------------------------------------
# Prediction rule
# ---------------------------------------------------------

# The threshold is the middle between the story average and non-story average
temporal_threshold = (story["temporal_per_100"] + nonstory["temporal_per_100"]) / 2

# Predict story if the text has more temporal markers than the threshold
predictions = []

for value in test_df["temporal_per_100"]:
    if value > temporal_threshold: # if a test has more than 1.2 temporal markers per 100 words predict story
        predictions.append("story")
    else:
        predictions.append("non-story")

test_df["prediction"] = predictions

# Calculate accuracy
final_accuracy = accuracy(test_df["prediction"], test_df["label"])


# Write required output file
with open("pragmatic_patterns.txt", "w", encoding="utf-8") as file:
    file.write("Pragmatic Patterns: Story vs Non-Story\n")
    file.write("======================================\n\n")

    file.write("Development data averages\n")
    file.write("-------------------------\n")
    file.write(averages.to_string())
    file.write("\n\n")

    file.write("Basic sentiment analysis\n")
    file.write("------------------------\n")
    file.write("Method: Asent and spacytextblob sentiment analysis.\n")
    file.write("Observation from development data:\n")

    file.write(
        f"Stories had an average negative sentiment score of "
        f"{story['negative_sentiment']:.3f}. "
        f"Non-stories had an average negative sentiment score of "
        f"{nonstory['negative_sentiment']:.3f}.\n"
    )

    file.write(
        f"Stories had an average neutral sentiment score of "
        f"{story['neutral_sentiment']:.3f}. "
        f"Non-stories had an average neutral sentiment score of "
        f"{nonstory['neutral_sentiment']:.3f}.\n"
    )

    file.write(
        f"Stories had an average positive sentiment score of "
        f"{story['positive_sentiment']:.3f}. "
        f"Non-stories had an average positive sentiment score of "
        f"{nonstory['positive_sentiment']:.3f}.\n"
    )

    file.write(
        f"Stories had an average compound sentiment score of "
        f"{story['compound_sentiment']:.3f}. "
        f"Non-stories had an average compound sentiment score of "
        f"{nonstory['compound_sentiment']:.3f}.\n"
    )

    file.write(
        f"Stories had an average subjectivity score of "
        f"{story['subjectivity']:.3f}. "
        f"Non-stories had an average subjectivity score of "
        f"{nonstory['subjectivity']:.3f}.\n"
    )

    file.write(
        "Rule: Sentiment scores were used as supporting pragmatic information, "
        "but not as the main prediction rule.\n"
    )

    file.write(
        "Explanation: Sentiment is useful for pragmatics because it shows emotion, "
        "attitude, and subjectivity. However, sentiment alone does not clearly separate "
        "stories from non-stories.\n\n"
    )


    file.write("Pattern 1: Emotional language\n")
    file.write("-----------------------------\n")
    file.write("Method: Count emotion words per 100 words.\n")
    file.write(f"Story average in development data: {story['emotion_per_100']:.3f}\n")
    file.write(f"Non-story average in development data: {nonstory['emotion_per_100']:.3f}\n")
    file.write(
        "Rule: Emotional language was used as a supporting pattern, "
        "but it was not used for the final prediction.\n"
    )
    file.write(
        "Explanation: This pattern may work because stories often describe personal or emotional "
        "experiences. It can fail because non-stories can also contain emotional opinions or complaints.\n\n"
    )


    file.write("Pattern 2: Temporal discourse markers\n")
    file.write("-------------------------------------\n")
    file.write("Method: Count temporal markers such as then, after, when, later, and finally.\n")
    file.write(f"Story average in development data: {story['temporal_per_100']:.3f}\n")
    file.write(f"Non-story average in development data: {nonstory['temporal_per_100']:.3f}\n")
    file.write(
        f"Rule applied to test data: If temporal_per_100 is higher than "
        f"{temporal_threshold:.3f}, predict story. Otherwise, predict non-story.\n"
    )
    file.write(f"Test accuracy: {final_accuracy:.3f}\n")
    file.write(
        "Explanation: This pattern works well because stories often describe events in a time "
        "sequence. It can fail when non-stories explain steps or processes in order.\n\n"
    )


    file.write("Pattern 3: Sentiment shift\n")
    file.write("--------------------------\n")
    file.write("Method: Split the text into beginning, middle, and end, then measure the change in sentiment.\n")
    file.write(f"Story average in development data: {story['sentiment_shift']:.3f}\n")
    file.write(f"Non-story average in development data: {nonstory['sentiment_shift']:.3f}\n")
    file.write(
        "Rule: Sentiment shift was used as a supporting pattern, "
        "but it was not used for the final prediction.\n"
    )
    file.write(
        "Explanation: This pattern may work because stories can develop emotionally over time. "
        "It can fail when a story has a stable tone or when a non-story changes tone strongly.\n\n"
    )


    file.write("Final prediction method\n")
    file.write("-----------------------\n")
    file.write(
        "The final prediction uses one simple pragmatic rule: temporal discourse markers. "
        "The development data showed that stories had more temporal markers than non-stories. "
        "Therefore, the system calculated the middle point between the story average and "
        "the non-story average. Test texts above this threshold were predicted as story. "
        "Test texts below this threshold were predicted as non-story.\n"
    )
    file.write(f"Final accuracy on the test data: {final_accuracy:.3f}\n\n")

    file.write("Conclusion\n")
    file.write("----------\n")
    file.write(
        "The most useful pragmatic pattern was temporal discourse markers, because stories "
        "often describe events in chronological order. Emotional language and sentiment "
        "shift were included as supporting pragmatic observations, but they were not used "
        "for the final prediction. Basic sentiment analysis was also included because it is "
        "relevant to pragmatics, but it did not clearly separate stories from non-stories."
    )


print("Created pragmatic_patterns.txt")
print(f"Final accuracy: {final_accuracy:.3f}")