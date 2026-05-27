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
                dep_by_label[label][token.dep_] += 1

for label in pos_by_label:
    total = sum(pos_by_label[label].values())

    print("\n" + label.upper() + " POS RATIOS")

    for pos, count in pos_by_label[label].most_common(10):
        ratio = count / total
        print(pos, round(ratio, 3))



for label in dep_by_label:
    total = sum(dep_by_label[label].values())

    print("\n" + label.upper() + " DEP RATIOS")

    for dep, count in dep_by_label[label].most_common(10):
        ratio = count / total
        print(dep, round(ratio, 3))

main_pos = ["PRON", "VERB", "AUX", "NOUN", "ADJ", "ADV", "PROPN"]

story_total = sum(pos_by_label["story"].values())
non_story_total = sum(pos_by_label["non-story"].values())

print("\nPOS COMPARISON")

for pos in main_pos:
    story_ratio = pos_by_label["story"][pos] / story_total
    non_story_ratio = pos_by_label["non-story"][pos] / non_story_total
    difference = story_ratio - non_story_ratio

    print(pos)
    print(" story:", round(story_ratio, 3))
    print(" non-story:", round(non_story_ratio, 3))
    print(" difference:", round(difference, 3))



main_deps = ["nsubj", "dobj", "pobj", "ROOT", "advmod", "amod", "compound", "prep"]
story_total = sum(dep_by_label["story"].values())
non_story_total = sum(dep_by_label["non-story"].values())

print("\nDEPENDENCY COMPARISON")

for dep in main_deps:
    story_ratio = dep_by_label["story"][dep] / story_total
    non_story_ratio = dep_by_label["non-story"][dep] / non_story_total
    difference = story_ratio - non_story_ratio

    print(dep)
    print(" story:", round(story_ratio, 3))
    print(" non-story:", round(non_story_ratio, 3))
    print(" difference:", round(difference, 3))

# print("STORY POS TAGS:")
# print(pos_by_label["story"])

# print("\nNON-STORY POS TAGS:")
# print(pos_by_label["non-story"])

# print("\nSTORY DEP TAGS:")
# print(dep_by_label["story"])

# print("\n NON STORY DEP TAGS:")
# print(dep_by_label["non-story"])
