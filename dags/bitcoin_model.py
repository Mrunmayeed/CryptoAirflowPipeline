import pandas as pd
from datetime import datetime
import pickle
import logging

from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler

from airflow.decorators import dag, task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

from preprocess_group import data_preprocess

logger = logging.getLogger(__name__)

@dag(dag_id='bitcoin_model_v1', start_date=datetime(2024, 4, 4))
def bitcoin_model():
    @task
    def scale_dataset(source_file, scaler_file):
        df = pd.read_csv(source_file)
        ss = StandardScaler()
        df['EWM']: ss.fit_transform(df['EWM'])
        pickle.dump(ss, open(scaler_file, 'wb'))
        dest_file = f'{source_file[:-4]}_scaled.csv'
        df.to_csv(dest_file, index=False)
        return dest_file

    @task
    def bitcoin_model(source_file, model_file):
        df = pd.read_csv(source_file)
        df = df.set_index('Date')
        model = ARIMA(df, order=(10, 1, 0))
        result_model = model.fit()
        pickle.dump(result_model, open(model_file, 'wb'))
        return model_file

    @task
    def upload_model(scaler_file,model_file):
        try:
            hook = S3Hook('bucket_conn')
            hook.load_file(
                filename=scaler_file,
                key=scaler_file,
                bucket_name='btc-dataset',
                replace=True
            )
            hook.load_file(
                filename=model_file,
                key=model_file,
                bucket_name='btc-dataset',
                replace=True
            )
        except Exception as e:
            logger.error(e)
            return e


    data_path='data/train/train.csv'
    scaler_file = 'data/scaler.pkl'
    model_file = 'data/model.sav'
    dest_file = data_preprocess(data_path)
    # dest_file = 'data/train/dataset_preprocessed_scaled.csv'
    dest_file = scale_dataset(source_file=dest_file, scaler_file=scaler_file)
    model_file = bitcoin_model(source_file=dest_file, model_file=model_file)
    upload_model(scaler_file=scaler_file, model_file=model_file)

bitcoin_model()