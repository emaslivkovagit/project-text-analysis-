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
    
#how many entities are there in story vs non-story text
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
    
    #nonstroy 
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
    print(f"Percent of Percentage, including ”%“ entities in story is {percent_story_p}")
    print(f"Percent of Percentage, including ”%“ entities in story is {percent_nonstory_p}")

# safe average used 
def the_avg(list):
    return round(sum(list) / len(list), 3) if list else 0

# Chain lenght
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
    print(f"average chains per 100 tokens in stories is {the_avg(nonstory)}")

# main
def main():
    filepath = sys.argv[1]
    rows = read_file(filepath)

    entity_count(rows)
    person_amount(rows) 
    chain_lenght(rows)
    chain_count(rows)

if __name__ == "__main__":
    main()