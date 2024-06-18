import os
import shutil

# Nazwa zestawu danych
dataset_name = 'galactus007/sms-smishing-collection-data-set'

# Pobierz zestaw danych i rozpakuj go
print("Downloading dataset...")
os.system(f'kaggle datasets download -d {dataset_name} --unzip -p smssmishcollection')

# Ścieżki plików i folderów
src_file = 'smssmishcollection/smssmishcollection/SMSSmishCollection.txt'
dst_file = './SMSSmishCollection.txt'
dst_folder = 'smssmishcollection'

# Sprawdź, czy plik SMSSmishCollection.txt istnieje wewnątrz folderu smssmishcollection
if os.path.exists(src_file):
    # Przenieś plik do katalogu nadrzędnego
    shutil.move(src_file, dst_file)
    print(f"Moved {src_file} to {dst_file}")

# Usuń folder smssmishcollection (zawiera dwa poziomy podkatalogu)
if os.path.exists(dst_folder):
    shutil.rmtree(dst_folder)
    print(f"Removed folder {dst_folder}")

print("Download completed.")
