#imports
import spacy
import coreferee
import nltk
import csv
from collections import Counter, defaultdict
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import sys

# loads english pipeline and wordnet
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("coreferee")

# read file
def read_file(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as file:
        read = csv.DictReader(file)
        for row in read:
            rows.append((row["content"], row["label"].strip().lower()))
        return rows
    
# how many entities are there in story vs non-story text
def entity_count(rows):
    storycount = 0
    noncount = 0
    story_texts = 0
    non_texts = 0

    for text, label in rows:
        doc = nlp(text)
        count = len(doc.ents)
        if label == "story":
            storycount += count
            story_texts += 1
        else:
            noncount += count
            non_texts += 1
    
    story_avg = storycount / story_texts if story_texts > 0 else 0
    non_avg = noncount / non_texts if non_texts > 0 else 0

    print("ENTITY COUNT")
    print(f"Average entities per story text: {story_avg}")
    print(f"Average entities per non-story text: {non_avg}")
    print("")

    return story_avg, non_avg

# Entity type in percentage 
def person_amount(rows):
    total_story = 0
    total_non = 0
    person_story = 0
    person_nonstory = 0
    norp_story = 0
    norp_nonstory = 0
    fac_story = 0
    fac_nonstory = 0
    org_story = 0
    org_nonstory = 0
    gpe_story = 0
    gpe_nonstory = 0
    product_story = 0
    product_nonstory = 0
    event_story = 0
    event_nonstory = 0
    lan_story = 0
    lan_nonstory = 0
    date_story = 0
    date_nonstory = 0
    time_story = 0
    time_nonstory = 0
    percent_story = 0
    percent_nonstory = 0

    ent_stats = {"story": Counter(), "non-story": Counter()}
    ent_totals = {"story": 0, "non-story": 0}

    for text, label in rows:
        doc = nlp(text)
        total = len(doc.ents)
       
        persons = 0
        norp = 0
        fac = 0
        org = 0
        gpe = 0
        product = 0
        event = 0
        lan = 0
        date = 0
        time = 0
        percent = 0
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                persons += 1
            if ent.label_ == "NORP":
                norp += 1
            if ent.label_ == "FAC":
                fac += 1
            if ent.label_ == "ORG":
                org += 1
            if ent.label_ == "GPE":
                gpe += 1
            if ent.label_ == "PRODUCT":
                product += 1
            if ent.label_ == "EVENT":
                event += 1
            if ent.label_ == "LANGUAGE":
                lan += 1
            if ent.label_ == "DATE":
                date += 1
            if ent.label_ == "TIME":
                time += 1
            if ent.label_ == "PERCENT":
                percent += 1

        if label == "story":
            total_story += total
            person_story += persons
            norp_story += norp 
            fac_story += fac
            org_story += org 
            gpe_story += gpe
            product_story += product
            event_story += event
            lan_story += lan
            date_story += date
            time_story += time
            percent_story += percent

        else:
            total_non += total
            person_nonstory += persons
            norp_nonstory += norp 
            fac_nonstory += fac
            org_nonstory += org 
            gpe_nonstory += gpe
            product_nonstory += product
            event_nonstory += event
            lan_nonstory += lan
            time_nonstory += time
            date_nonstory += date
            percent_nonstory += percent

        norm_label = label if label in ent_stats else "non-story"
        if total > 0:
            ent_totals[norm_label] += total
            for ent in doc.ents:
                ent_stats[norm_label][ent.label_] += 1

    #story 
    if total_story > 0:
        person_story_p = (person_story/total_story) * 100
        norp_story_p = (norp_story/total_story) * 100
        fac_story_p = (fac_story/total_story) * 100
        org_story_p = (org_story/total_story) * 100
        gpe_story_p = (gpe_story/total_story) * 100
        product_story_p = (product_story/total_story) * 100
        event_story_p = (event_story/total_story) * 100
        lan_story_p = (lan_story/total_story) * 100
        time_story_p = (time_story/total_story) * 100
        date_story_p = (date_story/total_story) * 100
        percent_story_p = (percent_story/total_story) * 100
    else:
        person_story_p = 0
        norp_story_p = 0
        fac_story_p = 0
        org_story_p = 0
        gpe_story_p = 0
        product_story_p = 0
        event_story_p = 0
        lan_story_p = 0
        time_story_p = 0
        date_story_p = 0
        percent_story_p = 0
    
    #nonstory 
    if total_non > 0:
        person_nonstory_p = (person_nonstory/total_non) * 100
        norp_nonstory_p = (norp_nonstory/total_non) * 100
        fac_nonstory_p = (fac_nonstory/total_non) * 100
        org_nonstory_p = (org_nonstory/total_non) * 100
        gpe_nonstory_p = (gpe_nonstory/total_non) * 100
        product_nonstory_p = (product_nonstory/total_non) * 100
        event_nonstory_p = (event_nonstory/total_non) * 100
        lan_nonstory_p = (lan_nonstory/total_non) * 100
        time_nonstory_p = (time_nonstory/total_non) * 100
        date_nonstory_p = (date_nonstory/total_non) * 100
        percent_nonstory_p = (percent_nonstory/total_non) * 100
    else:
        person_nonstory_p = 0
        norp_nonstory_p = 0
        fac_nonstory_p = 0
        org_nonstory_p = 0
        gpe_nonstory_p = 0
        product_nonstory_p = 0
        event_nonstory_p = 0
        lan_nonstory_p = 0
        time_nonstory_p = 0
        date_nonstory_p = 0
        percent_nonstory_p = 0

    print("Entity type in percentage")
    print("PERSON")
    print(f"Percent of person entities in story is {person_story_p}")
    print(f"Percent of person entities in non-story is {person_nonstory_p}")
    print("NORP")
    print(f"Percent of nationalities or religious or political groups entities in story is {norp_story_p}")
    print(f"Percent of nationalities or religious or political groups entities in non-story is {norp_nonstory_p}")
    print("FAC")
    print(f"Percent of buildings, airports, highways, bridges entities in story is {fac_story_p}")
    print(f"Percent of buildings, airports, highways, bridges entities in non-story is {fac_nonstory_p}")
    print(f"ORG")
    print(f"Percent of companies, agencies, institutions entities in story is {org_story_p}")
    print(f"Percent of companies, agencies, institutions entities in non-story is {org_nonstory_p}")
    print("GPE")
    print(f"Percent of countries, cities, states entities in story is {gpe_story_p}")
    print(f"Percent of countries, cities, states entities in non-story is {gpe_nonstory_p}")
    print("PRODUCT")
    print(f"Percent of objects, vehicles, foods, etc entities in story is {product_story_p}")
    print(f"Percent of objects, vehicles, foods, etc entities in non-story is {product_nonstory_p}")
    print("EVENT")
    print(f"Percent of named hurricanes, battles, wars, sports events, etc entities in story is {event_story_p}")
    print(f"Percent of named hurricanes, battles, wars, sports events, etc entities in non-story is {event_nonstory_p}")
    print("LANGUAGE")
    print(f"Percent of any named language entities in story is {lan_story_p}")
    print(f"Percent of any named language entities in non-story is {lan_nonstory_p}")
    print(f"DATE")
    print(f"Percent of absolute or relative dates or periods entities in story is {date_story_p}")
    print(f"Percent of absolute or relative dates or periods entities in non-story is {date_nonstory_p}")
    print("TIME")
    print(f"Percent of times smaller than a day entities in story is {time_story_p}")
    print(f"Percent of times smaller than a day entities in non-story is {time_nonstory_p}")
    print("PERCENT")
    print(f'Percent of Percentage, including "%" entities in story is {percent_story_p}')
    print(f'Percent of Percentage, including "%" entities in non-story is {percent_nonstory_p}')

    return ent_stats, ent_totals

# safe average used 
def the_avg(list):
    return round(sum(list) / len(list), 3) if list else 0

# Chain length
def chain_lenght(rows): 
    lenghts_story = []
    lengths_non = []

    for text, label in rows:
        doc = nlp(text)

        if doc._.coref_chains is None:
            continue
        if len(doc._.coref_chains) == 0:
            continue

        for chain in doc._.coref_chains:
            length = len(chain)
            if label == "story":
                lenghts_story.append(length) 

            if label == "non-story":
                lengths_non.append(length)
    
    print("")
    print("CHAIN LENGHT")
    print(f"The average chain lenght in stories is {the_avg(lenghts_story)}")
    print(f"The average chain lenght in non-stories is {the_avg(lengths_non)}")

    return the_avg(lenghts_story), the_avg(lengths_non)


# Chain count per 100 tokens
def chain_count(rows):
    story = []
    nonstory = []

    for text, label in rows:
        doc = nlp(text)
        token_count = len(doc)

        if token_count == 0:
            continue

        chain_count = 0
        if doc._.coref_chains:
            chain_count = len(doc._.coref_chains)
        else:
            chain_count = 0
        per_hundred = (chain_count/ token_count) * 100

        if label == "story":
            story.append(per_hundred)
        else: 
            nonstory.append(per_hundred)

    print("")
    print("CHAINS PER 100 TOKENS")
    print(f"average chains per 100 tokens in stories is {the_avg(story)}")
    print(f"average chains per 100 tokens in non-stories is {the_avg(nonstory)}")

    return the_avg(story), the_avg(nonstory)

def predict(test_rows, ent_stats, ent_totals,
            avg_chain_len_story, avg_chain_len_non,
            avg_chain_count_story, avg_chain_count_non,
            avg_entity_story, avg_entity_non):

    thr_entity_count = (avg_entity_story + avg_entity_non) / 2
    thr_chain_len = (avg_chain_len_story + avg_chain_len_non) / 2
    thr_chain_count = (avg_chain_count_story + avg_chain_count_non) / 2

    entity_types = ["PERSON", "NORP", "FAC", "ORG", "GPE", "PRODUCT", "EVENT", "LANGUAGE", "DATE", "TIME", "PERCENT"]

    def entity_type_p(label, etype):
        total = ent_totals[label]
        return (ent_stats[label][etype] / total * 100) if total > 0 else 0
    
    etype_thresholds = {}
    etype_story_higher = {}
    for etype in entity_types:
        story_percentage = entity_type_p("story", etype)
        nonstory_percentage = entity_type_p("non-story", etype)
        etype_thresholds[etype] = (story_percentage + nonstory_percentage) / 2
        etype_story_higher[etype] = story_percentage > nonstory_percentage
    
    story_higher_entity_count = avg_entity_story > avg_entity_non
    story_higher_chain_len = avg_chain_len_story > avg_chain_len_non
    story_higher_chain_count = avg_chain_count_story > avg_chain_count_non
    
    print("")
    print("THRESHOLDS LEARNED FROM DEV DATA")
    print(f"Entity count: story: {avg_entity_story} non-story: {avg_entity_non} threshold:{thr_entity_count}")
    print(f"Chain length: story: {avg_chain_len_story} non-story: {avg_chain_len_non} threshold: {thr_chain_len}")
    print(f"Chain count per 100 tokens: story: {avg_chain_count_story} non-story: {avg_chain_count_non} threshold: {thr_chain_count}")
    print("")
    print("Entity types:")
    for etype in entity_types:
        story_percentage = entity_type_p("story", etype)
        nonstory_percentage = entity_type_p("non-story", etype)
        print(f"{etype}: story: {story_percentage}% non-story: {nonstory_percentage}% threshold:{etype_thresholds[etype]}% (story is {'higher' if etype_story_higher[etype] else 'lower'})")

    # predicting using major vote
    correct = 0
    total_texts = 0

    for text, gold_label in test_rows:
        doc = nlp(text)
        token_count = len(doc)
        total_ents = len(doc.ents)
        votes_story = 0
        votes_non = 0 

        # entity count vote
        if story_higher_entity_count:
            if total_ents >= thr_entity_count:
                votes_story += 1
            else:
                votes_non += 1
        else:
            if total_ents <= thr_entity_count:
                votes_story +=1
            else: votes_non += 1
        
        # entity type vote (has 11 votes because multiple entity types)
        for etype in entity_types:
            if total_ents > 0:
                text_p = (sum(1 for e in doc.ents if e.label_ == etype) / total_ents) * 100
            else:
                text_p = 0
            if etype_story_higher[etype]:
                if text_p >= etype_thresholds[etype]:
                    votes_story += 1
                else:
                    votes_non += 1
            else:
                if text_p <= etype_thresholds[etype]:
                    votes_story += 1
                else:
                    votes_non += 1
        
        # chain length vote
        if doc._.coref_chains:
            avg_len = the_avg([len(chain) for chain in doc._.coref_chains])
        else:
            avg_len = 0
        if story_higher_chain_len:
            if avg_len >= thr_chain_len:
                votes_story += 1
            else:
                votes_non += 1
        else:
            if avg_len <= thr_chain_len:
                votes_story += 1
            else:
                votes_non += 1
        
        # chain per 100 tokens vote
        if token_count > 0:
            num_chains = len(doc._.coref_chains) if doc._.coref_chains else 0
            rate = (num_chains / token_count) * 100
        else:
            rate = 0
        if story_higher_chain_count:
            if rate >= thr_chain_count:
                votes_story += 1
            else:
                votes_non += 1
        else:
            if rate <= thr_chain_count:
                votes_story += 1
            else:
                votes_non += 1

        prediction = "story" if votes_story > votes_non else "non-story"
        if prediction == gold_label:
            correct += 1
        total_texts += 1

    accuracy = (correct / total_texts) * 100 if total_texts > 0 else 0
    print("")
    print("PREDICTION RESULTS ON TEST DATA")
    print(f"Correct: {correct}/{total_texts}")
    print(f"Accuracy: {accuracy}%")


# main
def main():
    dev_path = "dev_data/dev.csv"
    test_path = sys.argv[1]
    
    dev_rows = read_file(dev_path)
    test_rows = read_file(test_path)

    entity_count(test_rows)
    person_amount(test_rows) 
    chain_lenght(test_rows)
    chain_count(test_rows)

    avg_entity_story, avg_entity_non = entity_count(dev_rows)
    ent_stats, ent_totals = person_amount(dev_rows)
    avg_chain_len_story, avg_chain_len_non = chain_lenght(dev_rows)
    avg_chain_count_story, avg_chain_count_non = chain_count(dev_rows)

    predict(dev_rows, test_rows, ent_stats, ent_totals,
            avg_chain_len_story, avg_chain_len_non,
            avg_chain_count_story, avg_chain_count_non,
            avg_entity_story, avg_entity_non)

if __name__ == "__main__":
    main()