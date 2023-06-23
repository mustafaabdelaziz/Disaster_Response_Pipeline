# Disaster Response Pipeline Project

### Installation:
1. You have to install the `NLTK` library and also download the `punkt`, `wordnet`, and `stopwords`.
2. You have to import the `sklearn` library.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3000/

### Project Motivation:
This project is designed to classify the messages in order to identify Disaster related messages.

### File Description:
This file contains of many folders such as:
1. models folder which contains:
    - train_classifier.py: a Python script that builds a message classification Pipeline.
    - classifier.pkl: A saved classifier file.

2. data folder which contains:
    - disaster_categories.csv: A csv file that contains the categories of each message.
    - disaster_messages.csv: A csv file that contains each message and it's genre and id.
    - process_data.py: A Pyhton script that loads the data from csv files, merges them, cleans the data, and creates a DataBase.
    - DisasterResponse.db: A Database that contains the merged and cleaned data.

3. app folder which contains:
    - run.py: a Python script that uses Flask framework to create a backend system for the webapp.
    - templates: A folder that contains the html files for the webapp.
        * go.html: a HTML file that is used to deploy the results of the classification model.
        * master.html: a HTML file that containes the home page.

### Results:
Managed to create a classification model that is able to classify the message categories with high accuracy and also managed to deploy a webapp that can classify any message inputed to it.

### Licensing, Authors, Acknowledgements:
Must give credit to Stack Overflow for the data. You can find the Licensing for the data and other descriptive information at the Kaggle link available. Otherwise, feel free to use the code here as you would like!