Annotated Roc Story Data - JSON files

README contents:
A-Usage
B-JSON format (exact format definition)
C-Example

=============================================================================================
(A) Usage:

Annotations can be read via python:
   > import json
   > data = json.load("annotations.json")

#Access a story
   > idkey = "..."
   > story = data[idkey]

#Access story's info:
   > title = story["title"]                    # returns string with story title
   > train_dev_test = story["partition"]       # returns string: "train", "dev", or "test"
   > sentence1 = story["lines"]["1"]["text"]   # returns string: first line of story

#Access story line info:
   > line1 = story["lines"]["1"]
   > charsline = line1["characters"]    # dictionary of characters' motiv/emot annotations for 1st line
   > name = "..."
   > appears = charsline[name]["app"]   # boolean (whether they appear in this line)

#Annotations for a character on a specific line:
   > motivations = charsline[name]["motiv"]
   > freeresp_ann0 = motivations["ann0"]["text"]  # list of strings, freeresponses from annotator 0
   > plutchik_ann0 = charsline[name]["emotion"]["ann0"]["plutchik"]  # list of strings, plutchik categories from annotator 0 
                                                                     # (note: plutchik field only exists in dev/test set stories)


=============================================================================================
(B) JSON Format:

{
   $storyid_string: 
         {
            "partition": $train/dev/test_string,
            "storyid": $storyid_string,
            "title": $title_string,
            "lines": 
            {
               "1":
               {
                  "text": $sentence_string,
                  "characters": 
                  { 
                     $charactername:
                     {
                        "app":$boolean,
                        "motiv":
                        {
                           $hashed_workerid: {"text":[$opentext1, ...], "maslow":[$cat1, ...], "reiss":[$cat1, ...]},
                           ...
                        }
                        "emotion":
                        {
                           $hashed_workerid: {"text":[$opentext1, ...], "plutchik":[$cat1, ...]},
                           ...
                        }
                     }
                  }
               },

               "2": {...},
               "3": {...},
               "4": {...},
               "5": {...}
            }

         },

   $storyid2: ...
}

NOTES:
Maslow, reiss, and plutchik categories are not filled in for training data annotations
Motivation:
      - at test time we take the "majority label" for Maslow/Reiss to be categories voted on by > =2 workers
Emotion:
      - plutchik categories were rated on a three point scale and therefore are only listed if turkers rated them as 2 (moderate) or 3 (high), which is why they are listed as "joy:2" or "joy:3", etc.
      - at test time we take the "majority label" for Plutchik to be categories where worker average rating is > =2 (ex. one Turker did not select, one turker selectd Joy:2 and one turker selected Joy:3 -->  (1+2+3)/3 = 2)



=============================================================================================
(C) Example:
{
"0059a5df-25ab-4edc-86d1-44f7c5780876": {
    "partition": "dev", 
    "storyid": "0059a5df-25ab-4edc-86d1-44f7c5780876", 
    "title": "Amy get's lost"
    "lines": {
      "1": {
        "text": "Amy was going to visit her friend who just moved."
        "characters": {
          "Amy": {
            "app": true, 
            "emotion": {
              "ann0": {"plutchik": ["joy:3",  "anticipation:3"], "text": ["excited"]}, 
              "ann1": {"plutchik": ["joy:2", "anticipation:3"], "text": ["expectant"]}, 
              "ann2": {"plutchik": ["joy:2", "trust:2", "surprise:2", "anticipation:2"], "text": ["friendly"]}
            }, 
            "motiv": {
              "ann0": {"maslow": ["love"], "reiss": ["contact"], "text": ["to keep friendships"]}, 
              "ann1": {"maslow": ["love"], "reiss": ["contact"], "text": ["to be sociable"]},
              "ann2": {"maslow": ["love"], "reiss": ["contact"], "text": ["to prove her friendship"]}
            }
          }, 
          "Friend": {...}
          
        }
      },
      "2": {...}, 
      "3": {...}, 
      "4": {...},
      "5": {...}
    }
  }, 
   ...
}
