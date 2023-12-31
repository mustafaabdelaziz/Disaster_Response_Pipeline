import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Loading the messages and the categories datasets.

    Args:
        messages_filepath: the file path to the messages csv file.
        categories_filepath: the file path to the categories csv file.

    Returns:
        df: A dataframe that containes poth messages and categories data.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=["id"])
    return df


def clean_data(df):
    """ Cleaning the dataframe that containes messages and categories.

    Args:
        df: A dataframe that containes poth messages and categories data.

    Returns:
        df: A cleaned Dataframe the containes each category separated in a column.
    """
    # Naming the columns of the df with the names of the Categories
    categories = pd.Series(df['categories']).str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1, sort=False)
    df.drop_duplicates(inplace=True)
    # dropping wrong values
    df.drop(df[df['related'] == 2].index, inplace=True)
    return df


def save_data(df, database_filename):
    """ Saving the messages and categories DataFrame into a SQL DataBase.

    Args:
        df: A cleaned Dataframe the containes each category separated in a column.
        database_filename: The file path for the DataBase.

    Returns:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
