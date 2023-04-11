import pandas as pd


def load_rocstories(file_path):
    """Load data from a ROCStories CSV file into a DataFrame.

    Args:
        file_path (str): Path to the ROCStories CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path, delimiter=',',
                       usecols=["InputSentence1", "InputSentence2", "InputSentence3", "InputSentence4",
                                "RandomFifthSentenceQuiz1"])


def load_writingprompts(file_path):
    """Load data from a WritingPrompts CSV file into a DataFrame.

    Args:
        file_path (str): Path to the WritingPrompts CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path, delimiter=',', usecols=["wp1", "wp2", "wp3", "wp4", "wp5"])


def process_data(df, prefix):
    """Convert the DataFrame into a question-answer format.

    Args:
        df (pd.DataFrame): Input DataFrame with story sentences.
        prefix (str): Prefix to add to the column names.

    Returns:
        pd.DataFrame: Processed DataFrame with 'question' and 'answer' columns.
    """
    questions = []
    answers = []

    for index, row in df.iterrows():
        for i in range(len(row) - 1):
            question = " ".join([row[f"{prefix}{j + 1}"] for j in range(i)])
            questions.append(question)
            answers.append(row[f"{prefix}{i + 1}"])

    return pd.DataFrame({"question": questions, "answer": answers})


def main():
    rocstories_data = load_rocstories('./rocstories.csv')
    writingprompts_data = load_writingprompts('./writingprompts.csv')

    processed_rocstories = process_data(rocstories_data, "InputSentence")
    processed_writingprompts = process_data(writingprompts_data, "wp")

    combined_data = pd.concat([processed_rocstories, processed_writingprompts], ignore_index=True)
    combined_data.to_csv('combined_data.csv', index=False)


if __name__ == "__main__":
    main()
