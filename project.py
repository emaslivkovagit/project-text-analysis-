import csv
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_md")

pos_by_label = {
    "story": Counter(),
    "non-story": Counter()
}

dep_by_label = {
    "story": Counter(),
    "non-story": Counter()
}

with open("dev.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        text = row["content"]
        label = row["label"]

        doc = nlp(text)

        for token in doc:
            if not token.is_punct and not token.is_space:
                pos_by_label[label][token.pos_] += 1

        for token in doc:
            if not token.is_punct and not token.is_space:
                dep_by_label[label][token.dep] += 1


print("STORY POS TAGS:")
print(pos_by_label["story"])

print("\nNON-STORY POS TAGS:")
print(pos_by_label["non-story"])

print("STORY DEP TAGS:")
