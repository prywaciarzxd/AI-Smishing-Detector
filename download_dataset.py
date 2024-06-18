import os
import shutil

# Nazwa zestawu danych
dataset_name = 'galactus007/sms-smishing-collection-data-set'

# Pobierz zestaw danych i rozpakuj go
os.system(f'kaggle datasets download -d {dataset_name} --unzip -p smssmishcollection')

# Ścieżki plików i folderów
src_file = 'smssmishcollection/SMSSmishCollection.txt'
dst_file = './SMSSmishCollection.txt'
dst_folder = 'smssmishcollection'

# Przenieś plik do katalogu nadrzędnego
if os.path.exists(src_file):
    shutil.move(src_file, dst_file)
    print(f"Moved {src_file} to {dst_file}")

# Usuń nowo utworzony folder
if os.path.exists(dst_folder):
    shutil.rmtree(dst_folder)
    print(f"Removed folder {dst_folder}")

print("Download completed.")
