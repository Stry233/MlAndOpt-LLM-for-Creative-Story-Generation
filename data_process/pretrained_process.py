# Import required libraries
import ast

import pandas as pd
import sng_parser
from pandas import DataFrame

# Specify global variables
batch_size = 26
num_epochs = 20
fileTag = "original-plutchik-v1"

# Read data from CSV files
trainDatasetOriginal = pd.read_csv(
    f'./A-Emotion-Data-Cleaning-Pipeline-for-Enhancing-Data-Reliability/data/csv_version/dev/emotion/allcharlinepairs-{fileTag}.csv')
testDatasetOriginal = pd.read_csv(
    f'./A-Emotion-Data-Cleaning-Pipeline-for-Enhancing-Data-Reliability/data/csv_version/test/emotion/allcharlinepairs-{fileTag}.csv')

# Combine and shuffle datasets
trainDatasetOriginal = pd.concat([trainDatasetOriginal, testDatasetOriginal]).reset_index(drop=True)


# Function to extract keywords from a sentence
def getKeyword(inputSentence: str) -> str:
    """
    Extracts the main keywords from a given sentence.

    :param inputSentence: The input sentence as a string.
    :return: A string containing the extracted keywords, separated by commas.
    """
    graph = sng_parser.parse(inputSentence)
    return (", ".join((key.lower()).replace("a ", "").replace("the ", "").replace("an ", "")
                      for key in [x['span'] for x in graph['entities']])).replace("i,", "I,")


# Function to parse and remap Plutchik emotions
def parseRemapPlutchik(emotionList: str) -> str:
    """
    Parses and remaps the Plutchik emotions from a given emotion list string.

    :param emotionList: The input emotion list as a string.
    :return: A string containing the parsed and remapped Plutchik emotions, separated by commas.
    """
    emotionListDict = ast.literal_eval(emotionList)
    report = ""
    for emoLabel in ['joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation']:
        report += "" if emotionListDict[emoLabel] == 0 else f"{emoLabel}, "
    return report.strip(", ")


# Function to generate questions with the previous sentence and prompt
def getPreviousSentenceWPrompt(inputDf: pd.DataFrame) -> list:
    """
    Generates a list of questions by extracting previous sentences and prompts from the input DataFrame.

    :param inputDf: The input DataFrame containing the sentences and prompts.
    :return: A list of questions generated from the input DataFrame.
    """
    # Filters
    previousSentenceFilter = inputDf[[
        True if index != len(inputDf['linenum']) - 1 and inputDf.iloc[index]['linenum'] < inputDf.iloc[index + 1][
            'linenum'] else False for index, linenum in
        enumerate(inputDf['linenum'])]]
    nextEmotionFilter = inputDf[
        [True if index != 0 and inputDf.iloc[index]['linenum'] > inputDf.iloc[index - 1]['linenum'] else False for
         index, linenum in enumerate(inputDf['linenum'])]]

    # Generate questions
    return [(f"Generate next sentence that makes reader feels {parseRemapPlutchik(nextEmotion[1]['plutchik'])}."
             + f"<extra_id_0>KEYWORD: {getKeyword(nextEmotion[1]['sentence'])}"
             + f"<extra_id_1>CONTEXT: {nextEmotion[1]['context'].replace('|', '')}")
            for previousSentence, nextEmotion in zip(previousSentenceFilter.iterrows(), nextEmotionFilter.iterrows())]


# Function to get the next sentence
def getNextSentence(inputDf: pd.DataFrame) -> pd.Series:
    """
    Extracts the next sentences from the input DataFrame.

    :param inputDf: The input DataFrame containing the sentences.
    :return: A pandas Series containing the next sentences.
    """
    return inputDf[
        [True if index != 0 and inputDf.iloc[index]['linenum'] > inputDf.iloc[index - 1]['linenum'] else False for
         index, linenum in enumerate(inputDf['linenum'])]]['sentence']


# Process data
processedDataTrain = DataFrame({'question': getPreviousSentenceWPrompt(trainDatasetOriginal),
                                "answer": getNextSentence(trainDatasetOriginal)}).reset_index(drop=True)
processedDataTest = DataFrame({'question': getPreviousSentenceWPrompt(testDatasetOriginal),
                               "answer": getNextSentence(testDatasetOriginal)}).reset_index(drop=True)

processedDataTrain.to_csv(f'./genV2-{fileTag}-train.csv', index=False)
processedDataTest.to_csv(f'./genV2-{fileTag}-test.csv', index=False)