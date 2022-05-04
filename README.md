# Gonzaga CPSC322: Data Science Algorithms Final Project
This project uses the "3-Year Recidivism for Offenders Released from Prison in Iowa" dataset from data.iowa.gov found [here](https://data.iowa.gov/Correctional-System/3-Year-Recidivism-for-Offenders-Released-from-Pris/mw8r-vqy4)

Contributors: [@EricGustin](https://github.com/EricGustin) and [@danHoberman1999](https://github.com/danHoberman1999)

## How to use the Flask API:
1. `python prediction_api.py` from commandline
2. Open a browser and go to localhost
3. The url`http://localhost:PORT/predictor` should be given a query string that will be used to predict using a trained Random Forest Classifier. An example url would be `http://localhost:5001/predictor?0=2010.0&1=2013.0&2=NA&3=Parole&4=Black+-+Non-Hispanic&5=25-34&6=Male&7=C+Felony&8=Violent&9=Robbery`
4. The classifier's prediction will be displayed in the browser as JSON

## Important Files:
* `trained_random_forest.pkl` is our trained & pickled Random Forest Classifier object
* `test_myclassifiers.py` contains tests for our classifiers
* `technical_report.ipynb`describes our approach and findings
* `random_forest_pickler.py` trains a Random Forest Classifier on all data and then pickles it to a file
* `project_proposal.ipynb` is a document describe and proposal the dataset that we used for this project
* `prediction_api.py` is our Flask API for predicting a test instance
* `plot_utils.py` contains helper functions for our EDA
* `exploratory_data_analysis.ipynb` is a document containing varying charts and graphs that give us a better understanding of the recidivism dataset
* `data_cleaning.py` is a file that cleans our dataset and writes the cleaned data to file
* `mysklearn/` contains our mysklearn module including four different classifiers, evaluation metrics, `MyPyTable` and helper functions for our classifiers
* `data/3-Year_Recidivism_for_Offenders_Released_from_Prison_in_Iowa.csv` is the dataset before being cleaned
* `data/cleaned-recidivism-data-NA.csv` is our cleaned dataset that we use for training and testing
