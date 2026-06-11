import csv
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_md")

# Analyze one dataset file and collect syntax-based patterns and rule results.

def analyze_dataset(filename):
    """Set up counters to store syntax patterns and 
    rule evaluation results for story and non-story texts.
    """
     

    pos_by_label = {
        "story": Counter(),
        "non-story": Counter()
    }

    dep_by_label = {
        "story": Counter(),
        "non-story": Counter()
    }

    correct_predictions = 0
    total_texts = 0
    prediction_results = Counter()

    correct_vbd_predictions = 0
    total_vbd_texts = 0
    vbd_prediction_results = Counter()

    vbd_by_label = {
        "story": 0,
        "non-story": 0
    }

    token_total_by_label = {
        "story": 0,
        "non-story": 0
    }

    
    #open dataset and loops through rows in the file
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            text = row["content"]
            label = row["label"]

            doc = nlp(text)

            text_pos = Counter()
            token_count = 0
            text_vbd_count = 0

            # Clean up tokens updates counts
            for token in doc:
                if not token.is_punct and not token.is_space:
                    pos_by_label[label][token.pos_] += 1
                    dep_by_label[label][token.dep_] += 1
                    token_total_by_label[label] += 1

                    text_pos[token.pos_] += 1
                    token_count += 1
                    if token.tag_ == "VBD":
                        vbd_by_label[label] += 1
                        text_vbd_count += 1

            # Ratio for PRON and ADV rule
            if token_count > 0:
                pron_ratio = text_pos["PRON"] / token_count
                adv_ratio = text_pos["ADV"] / token_count
                story_score = pron_ratio + adv_ratio

                # Update prediction counts
                if story_score > 0.19:
                    prediction = "story"
                else:
                    prediction = "non-story"

                total_texts += 1

                # Updates the label count
                if prediction == label:
                    correct_predictions += 1

                prediction_results[(label, prediction)] += 1
        
                # Ratio for VBD Rule
                vbd_ratio = text_vbd_count / token_count

                # Actual VBD Rule
                if vbd_ratio > 0.03:
                    vbd_prediction = "story"
                else:
                    vbd_prediction = "non-story"

                total_vbd_texts += 1

                
                # Updates VBD Count
                if vbd_prediction == label:
                    correct_vbd_predictions += 1

                vbd_prediction_results[(label, vbd_prediction)] += 1


    return (
    pos_by_label,
    dep_by_label,
    correct_predictions,
    total_texts,
    prediction_results,
    vbd_by_label,
    token_total_by_label,
    correct_vbd_predictions,
    total_vbd_texts,
    vbd_prediction_results
) 

def print_ratios(counter_by_label, title, top_n=10):
    """Print the most common syntax tags as ratios for each label.
    """

    # Total number of labels  / tokens
    for label in counter_by_label:
        total = sum(counter_by_label[label].values())

        print(f"\n{label.upper()} {title} RATIOS")

        # Divides tag count by total number of tokens in that label
        for item, count in counter_by_label[label].most_common(top_n):
            ratio = count / total
            print(item, round(ratio, 3))

def print_comparison(counter_by_label, items, title):
    """Compare selected POS or dependency tags between story and non-story.
    """

    # Calculates total story / non-story tokens
    story_total = sum(counter_by_label["story"].values())
    non_story_total = sum(counter_by_label["non-story"].values())

    print(f"\n{title} COMPARISON")

    # Loops through items and divides by total number of tokens
    for item in items:
        story_ratio = counter_by_label["story"][item] / story_total
        non_story_ratio = counter_by_label["non-story"][item] / non_story_total
        difference = story_ratio - non_story_ratio

        print(item)
        print(" story:", round(story_ratio, 3))
        print(" non-story:", round(non_story_ratio, 3))
        print(" difference:", round(difference, 3))

def print_vbd_results(vbd_by_label, token_total_by_label):
    """
    Print the ratio of past-tense verbs for story and non-story texts.
    """
    print("\nPAST-TENSE VERB PATTERN")

    # Loops through labels in the variable and returns the ratio
    for label in vbd_by_label:
        ratio = vbd_by_label[label] / token_total_by_label[label]
        print(label, "VBD ratio:", round(ratio, 3))


def print_rule_results(title, rule_description, correct, total, results):
    """
    Print the accuracy and prediction results for one rule.
    """
    print("\n" + title)
    print("Rule:", rule_description)
    print("Correct:", correct)
    print("Total texts:", total)

    if total > 0:
        print("Accuracy:", round(correct / total, 3))
    else:
        print("Accuracy: no texts counted")

    print("Prediction results:")
    print(results)


(
    pos_by_label,
    dep_by_label,
    correct_predictions,
    total_texts,
    prediction_results,
    vbd_by_label,
    token_total_by_label,
    correct_vbd_predictions,
    total_vbd_texts,
    vbd_prediction_results
) = analyze_dataset("../dev_data/test.csv")

main_pos = ["PRON", "VERB", "AUX", "NOUN", "ADJ", "ADV", "PROPN"]
main_deps = ["nsubj", "dobj", "pobj", "ROOT", "advmod", "amod", "compound", "prep"]

print_ratios(pos_by_label, "POS")
print_ratios(dep_by_label, "DEP")

print_comparison(pos_by_label, main_pos, "POS")
print_comparison(dep_by_label, main_deps, "DEPENDENCY")

print_vbd_results(vbd_by_label, token_total_by_label)

print_rule_results(
    "SIMPLE PRON + ADV RULE",
    "if PRON ratio + ADV ratio > 0.19, predict story",
    correct_predictions,
    total_texts,
    prediction_results
)

print_rule_results(
    "SIMPLE VBD RULE",
    "if VBD ratio > 0.03, predict story",
    correct_vbd_predictions,
    total_vbd_texts,
    vbd_prediction_results
)

with open("../syntax_patterns.txt", "w", encoding="utf-8") as file:

    file.write("""Syntax Patterns: Story vs Non-Story\n

Pattern 1: Pronouns and adverbs

In the development data, story texts used slightly more pronouns and adverbs than non-story texts. I checked this with spaCy POS tags, using token.pos_.

The idea was that stories are often more personal. They often use words like “I”, “he”, “she”, “they”, and adverbs like “then”, “suddenly”, or “really”.

Rule used:
If the PRON ratio + ADV ratio was higher than 0.19, the text was classified as story. Otherwise, it was classified as non-story.

Result:
The rule got 212 out of 351 texts correct. The accuracy was 0.604.

This pattern worked a bit, but it was not very strong. It failed when non-story posts were also written in a personal style, or when stories did not use many pronouns or adverbs.

Pattern 2: Past-tense verbs

The stronger pattern was past-tense verbs. Story texts had more VBD tags than non-story texts. I checked this with spaCy’s fine-grained POS tags, using token.tag_.

The VBD ratio was 0.04 for story texts and 0.02 for non-story texts. This means that stories used about twice as many past-tense verbs.

Rule used:
If the VBD ratio was higher than 0.03, the text was classified as story. Otherwise, it was classified as non-story.

Result:
The rule got 235 out of 351 texts correct. The accuracy was 0.67.

This worked better because many stories describe things that happened in the past. It failed when a story was not written in past tense, or when a non-story text also talked about past events.

Additional dependency observation

I also looked at dependency labels with token.dep_. These differences were smaller. Story texts had slightly more advmod and dobj labels. Non-story texts had slightly more compound, amod, and prep labels.

This was not a strong pattern. The sentence structures of story and non-story Reddit posts were often quite similar. For this dataset, POS tags and past-tense verbs were more useful than dependency labels.
""")