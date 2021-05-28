import datetime
import pytz
def dtnow(tz="America/New_York"):
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    pst_now = utc_now.astimezone(pytz.timezone("America/New_York"))
    return pst_now.strftime("%Y%m%d%H%M")

from google.cloud import storage

project = "strategic-howl-305522"
bucket_name = "gcs-station-168"           
storage_client = storage.Client(project=project)
bucket = storage_client.bucket(bucket_name)

# # GCS download
import pickle
remote_filename = 'gcs_models.txt' # [1]
source_blob_name = f"tps-may-2021-label/{remote_filename}" # (2) No prefix slash
destination_file_name = f"/kaggle/working/{remote_filename}" # (3)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)
import pandas as pd

# GCS upload best model & note
best_score = 1.012 # float [1]
architecture = 'lgbm' # [2]
model_note = '8fold2repeat' # [3]
local_folder = '/kaggle/working/' # [4]
gcs_folder = 'tps-may-2021-label/' # [5]
dtnow = dtnow()
local_filename = f'{dtnow}-{best_score:.5f}-{architecture}-{model_note}.pickle'
pickle.dump(best_model, open(f'{local_folder}{local_filename}', 'wb')) # [6]
blob = bucket.blob(f'{gcs_folder}{local_filename}')
blob.upload_from_filename(f'{local_folder}{local_filename}')   
print(f'local: {local_folder}{local_filename}')
print(f'remote: {gcs_folder}{local_filename}')