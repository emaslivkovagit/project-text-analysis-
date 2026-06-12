# project-text-analysis-

## Instructions on how to run semantics.py
To run the script on the development data run:
python3 semantics.py dev_data/dev.csv > semantic_patterns_dev.txt
To run the script on the test data run:
python3 semantics.py dev_data/test.csv > semantic_patterns_test.txt

Sammary Semantics (author Vera Smid):
The sematics program looks at namend entities and chain lenghts to find patterns in the development
data. It looks at "PERSON", "NORP", "FAC", "ORG", "GPE", "PRODUCT", "EVENT", "LANGUAGE", "DATE", "TIME", "PERCENT" to see if there is a difference in the amount of appearences between a story or non-story text. It also looks at chain lenghts and the average amount of entities in a story vs non-story text. 

## Instructions on how to run pragmatics_analysis.py 
python3 analysis/pragmatics_analysis.py dev_data/dev.csv dev_data/test.csv

Summary Pragmatics (author Ema Slivkova):
The program analyses texts from a development and test CSV to find pragmatic differences between stories and non-stories. It calculates sentiment scores, emotional words, temporal markers, and sentiment shift for each text.
Then it uses the development data to see which patterns differ between stories and non-stories. For the final prediction, it uses one simple rule: texts with more temporal discourse markers, such as then, after, later, and finally, are predicted as stories. It then checks this rule on the test data and writes the results to pragmatic_patterns.txt.

## Instructions on how to run syntax_analysis.py
To run  the scrip on the development data run:
python analysis/syntax_analysis.py dev_data/dev.csv
To run  the scrip on the test data run:
python analysis/syntax_analysis.py dev_data/test.csv 

Summary Syntax (author Victor Altilio):
The syntax script looks at differences between story and non-story texts using spaCy. It checks POS tags such as PRON, ADV, VERB, NOUN and ADJ, and compares how often they appear in both groups. Based on the development data, I used a simple PRON + ADV rule, because stories had slightly more pronouns and adverbs.

The script also checks VBD tags, which are past-tense verbs like “went”, “said” or “happened”. This was the clearest syntax pattern, because stories often describe things that happened in the past. I also compared dependency labels such as nsubj, dobj, advmod, compound and amod, but these differences were smaller. Because of that, dependency labels were mainly used for comparison, not as the main rule. The script prints the POS ratios, dependency ratios, story/non-story comparisons, and the results of the PRON + ADV and VBD rules.