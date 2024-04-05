import pandas as pd
from datetime import datetime
import logging

from airflow.decorators import task_group, task
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

logger = logging.getLogger(__name__)

@task_group(group_id='data_preprocess')
def data_preprocess(data_path):

    @task
    def download_dataset(source_file, dest_file):
        hook = S3Hook('bucket_conn')
        file_name = hook.download_file(
            key=source_file,
            bucket_name='btc-dataset',
            local_path=dest_file
        )

        # will return absolute path
        return dest_file

    @task
    def preprocess_dataset(source_file):
        df = pd.read_csv(source_file)
        df = df.set_index('Date').sort_values('Date').drop(columns=['Unix Timestamp', 'Symbol'])
        df['Avg'] = df[['High', 'Low']].mean(axis=1)
        df['EWM'] = df.Avg.ewm(alpha=0.005).mean()
        df = df[['EWM']].reset_index()
        dest_file = f'{source_file[:-4]}_preprocessed.csv'
        df.to_csv(dest_file, index=False)
        return dest_file


    intermediate = 'data/train/' if 'train' in data_path else 'data/test/'
    dest_file = download_dataset(source_file=data_path, dest_file=intermediate)
    dest_file = preprocess_dataset(source_file=dest_file)
    return dest_file

