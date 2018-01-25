import pandas as pd


def process_data(path):
    '''
    Process agnews data set to fit the overall format of the project.
    :param path: path of an agnews dataset csv
    :return: pandas dataframe
    '''
    df = pd.read_csv(path, names=["class", "title", "comment"])
    df["text"] = df["title"] + " " + df["comment"]

    df = df.drop(["title", "comment"], axis=1)

    return df
