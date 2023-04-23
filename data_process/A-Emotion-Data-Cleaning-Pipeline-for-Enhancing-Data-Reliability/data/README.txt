Annotated Roc Story Data
(ACL 2018)

Readme contents:
(A) JSON vs. CSV
(B) File Listing
(C) Annotation Description
(D) Stats

-----------------------------------------------------------
(A) JSON vs. CSV:
The two directories contain differently formatted versions of the same data. The JSON is more compact, but the CSV files may be easier to use with data-readin packages like Pandas.
For more details about the formatting of each, see the README in each directory.

------------------------------------------------------------
(B) File Listing:

   Story List:
   (1) rocstorysubset.csv: ~15000 stories pulled randomly from the roc story training set (train 10k, dev 2.5k, test 2.5k)
   (2) storyid_partition.txt: train/dev/test partition for each storyid

   Data:
   (1) json_version: json formatting --> see json_version/README.txt 
   (2) csv_version: csv formatting --> see csv_version/README.txt 

------------------------------------------------------------
(C) Annotation Description: 
   For this project we annotated character motivations and emotions within the context of stories as they change per line.  In addition to free responses, we also asked for categorical responses using commonly used psychological categorizations (maslow, reiss, plutchik).
   Due to the expense of getting fine-grained annotations per character per line, we used two pipelines, one with only open-text annotations (training) and one with more detailed categorical annotations (dev/test). 

   First, all stories were preprocessed for lists of characters appearing in the stories using this HIT along with a list of lines in which characters appear.

   Then the story was annotated according to one of two pipelines:
      -Sparse Pipeline (Train) : One in which we just collect open-text responses
      -Fine Grained Pipeline (Dev/Test): A longer annotation pipeline where we also get more fine grained categorical labels

------------------------------------------------------------
(D) Stats:
Motivation Categories:
Maslow: (5 category) spiritual growth, esteem, love, stability, physiological
Reiss: (19 category) curiosity, serenity, idealism, independence,
            competition, honor, approval, power, status, 
            romance, belonging, family, social contact, 
            health, savings, order, safety, 
            food, rest

Emotional Catgories:
Plutchik: (8 category) joy, trust, fear, surprise, sadness, disgust, anger, anticipation
Annotators who selected an emotional category either selected 3 (high) or 2 (moderate), so they are all annotated as "joy:2" or "joy:3", etc.


There are 14738 annotated stories in the final train/dev/test splits that we used for testing in the paper (we started with 15000 stories randomly selected from ROC stories training set and split as 10k/2.5k/2.5k across our train/dev/test partitions, though some are not included in the final dataset for lack of characters or annotation quality reasons):

                         | Train  |  Dev   |  Test
             ---------------------------------------
             # stories   | 9885   | 2483   | 2370
# character-line pairs:
     ... w/ motivation   | 40154  | 8762   | 6831
        ... w/ emotion   | 76613  | 14532  | 13785

