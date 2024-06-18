import subprocess

if __name__ == "__main__":

    #Pobieranie datasetu z kaggle
    print("Uruchamianie download_dataset.py...")
    subprocess.run(["python3", "download_dataset.py"])
    print("download_dataset.py zakończony.\n")
    # Uruchomienie remove_duplicates.py
    print("Uruchamianie remove_duplicates.py...")
    subprocess.run(["python3", "remove_duplicates.py"])
    print("remove_duplicates.py zakończony.\n")

    # Uruchomienie tokenik.py
    print("Uruchamianie tokenik.py...")
    subprocess.run(["python3", "tokenik.py"])
    print("tokenik.py zakończony.\n")

    # Uruchomienie message_label.py
    print("Uruchamianie message_label.py...")
    subprocess.run(["python3", "message_label.py"])
    print("message_label.py zakończony.\n")

    # Uruchomienie classifiers.py
    print("Uruchamianie classifiers.py...")
    subprocess.run(["python3", "classifiers.py"])
    print("classifiers.py zakończony.\n")

    print("Wszystkie skrypty zostały pomyślnie uruchomione i zakończone.")
