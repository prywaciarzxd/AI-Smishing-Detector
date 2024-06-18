import os


dataset_name = 'galactus007/sms-smishing-collection-data-set'

os.system(f'kaggle datasets download -d {dataset_name} --unzip')

print("Download completed.")
