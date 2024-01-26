


import pickle
import logging
from datetime import date


import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


logger = logging.getLogger(__name__)



def read_dataframe(filename) -> pd.DataFrame:
    """
    Read a Parquet file and preprocess the dataframe.

    This function reads a Parquet file into a pandas DataFrame, computes the
    duration of the trips, filters the trips based on duration, and converts
    categorical columns to string type.

    Parameters:
    filename (str): The path to the Parquet file to be read.

    Returns:
    pandas.DataFrame: A preprocessed DataFrame with the trip data.
    """ 
    try:
        logger.info(f"reading data from {filename}")

        df = pd.read_parquet(filename)
        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)      
        return df
 
    except Exception as e:
        logger.error(f"Error reading {filename}: {e}")
        raise

def train(train_month: date, val_month: date, model_output_path:str) -> None:
    """
    Train a linear regression model on taxi trip data.

    This function trains a linear regression model to predict the duration of taxi trips.
    The model is trained using data from a specified training month and validated
    on data from a specified validation month. The trained model is then saved to a specified path.

    Parameters:
    train_month (date): The month for which training data should be used.
    val_month (date): The month for which validation data should be used.
    model_output_path (str): The path where the trained model will be saved.

    Returns:
    None
    """

    
    url_template = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    train_url = url_template.format(year=train_month.year, month=train_month.month)
    val_url = url_template.format(year=val_month.year, month=val_month.month)

    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)


    pipeline = make_pipeline(
        DictVectorizer(),
        LinearRegression()
    )


    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    logger.debug(f"turning dataframes into dictionaries...") 

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    logger.debug(f"number of records in train: {len(train_dicts)}")
    logger.debug(f"number of records in validation {len(val_dicts)} ")

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    logger.debug(f"training the model...")


    pipeline.fit(train_dicts, y_train)

    y_pred = pipeline.predict(val_dicts)

    rmse = mean_squared_error(y_val, y_pred, squared=False)

    logger.info(f"rmse = {rmse}")

    logger.info(f"saving the model to {model_output_path}...")


    with open(model_output_path, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
