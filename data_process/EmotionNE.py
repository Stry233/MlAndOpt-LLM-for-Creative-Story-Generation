import pandas as pd
import sng_parser
from tqdm import tqdm
from transformers import pipeline

emotion_classifier = "Yuetian/roberta-large-finetuned-plutchik-emotion"


def load_data(file_path):
    """Load data from a CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path)


def create_context_all_column(df):
    """Create a new column 'context_all' by concatenating 'question' and 'answer' columns.

    Args:
        df (pd.DataFrame): DataFrame containing 'question' and 'answer' columns.

    Returns:
        pd.DataFrame: DataFrame with the new 'context_all' column added.
    """
    df["context_all"] = df["question"] + " " + df["answer"]
    return df


def add_emotion_column(df, input_column, output_column):
    """Add a new column with emotion classification results for the given input column.

    Args:
        df (pd.DataFrame): DataFrame containing the input column.
        input_column (str): Column name with text data to classify.
        output_column (str): Column name to store the classification results.

    Returns:
        pd.DataFrame: DataFrame with the new output column added.
    """
    classifier = pipeline("text-classification", model=emotion_classifier, tokenizer=emotion_classifier, top_k=None,
                          device=0)
    tqdm.pandas(desc="Processing")
    df[output_column] = df[input_column].progress_apply(classifier)
    return df


def genKeyword(demoSentenceInput):
    """Extract keywords from a given input text.

    Args:
        demoSentenceInput (str): Input text to extract keywords from.

    Returns:
        str: Comma-separated keywords.
    """
    graph = sng_parser.parse(demoSentenceInput)
    majorKeyword = [x['span'] for x in graph['entities']]
    return ", ".join(key for key in majorKeyword)


def add_keyword_column(df, input_column, output_column):
    """Add a new column with keywords extracted from the given input column.

    Args:
        df (pd.DataFrame): DataFrame containing the input column.
        input_column (str): Column name with text data to extract keywords from.
        output_column (str): Column name to store the extracted keywords.

    Returns:
        pd.DataFrame: DataFrame with the new output column added.
    """
    tqdm.pandas(desc="Processing")
    df[output_column] = df[input_column].progress_apply(genKeyword)
    return df


def get_high_score_labels(emotions_list):
    """Filter out labels with scores below 0.5 from an emotion classification result.

    Args:
        emotions_list (list): List of emotion classification results, where each result is a dictionary.

    Returns:
        list: Filtered emotion labels with scores above or equal to 0.5.
    """
    return [item['label'] for item in emotions_list[0] if item['score'] >= 0.5]


def add_filtered_emotions_column(df, input_column, output_column):
    """Add a new column with filtered emotion labels from the given input column.

    Args:
        df (pd.DataFrame): DataFrame containing the input column.
        input_column (str): Column name with emotion classification results.
        output_column (str): Column name to store the filtered emotion labels.

    Returns:
        pd.DataFrame: DataFrame with the new output column added.
    """
    df[output_column] = df[input_column].apply(get_high_score_labels)
    return df


def save_data(df, file_path):
    """Save DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to the CSV file to save the DataFrame.
    """
    df.to_csv(file_path, index=False)


def main():
    """Main function to run the data processing steps in sequence."""
    storyData = load_data('./ROC_clean.csv')
    storyData = create_context_all_column(storyData)
    storyData = add_emotion_column(storyData,
                                   input_column="context_all",
                                   output_column="emotion")
    storyData = add_keyword_column(storyData,
                                   input_column="answer",
                                   output_column="keywords")
    storyData = add_filtered_emotions_column(storyData,
                                             input_column="emotion",
                                             output_column="filtered_emotions")
    save_data(storyData, 'ROC_clean.csv')


if __name__ == "__main__":
    main()

