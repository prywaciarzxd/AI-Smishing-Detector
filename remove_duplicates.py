import argparse
import re
from collections import defaultdict
import os

# Funkcja normalizująca tekst
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\d', 'N', text)  # Zamiana cyfr na 'N'
    text = re.sub(r'\W+', ' ', text)  # Zamiana znaków specjalnych na białe spacje
    return text.strip()

# Funkcja generująca N-gramy z tekstu
def generate_ngrams(text, n):
    words = text.split()
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

# Funkcja usuwająca duplikaty z listy wiadomości
def remove_duplicates(data, n_values):
    seen_ngrams = defaultdict(set)
    unique_messages = []

    for line in data:
        label, text = line.split('\t')
        normalized_text = normalize_text(text)
        is_duplicate = False
        for n in n_values:
            ngrams = generate_ngrams(normalized_text, n)
            for ngram in ngrams:
                if ngram in seen_ngrams[n]:
                    is_duplicate = True
                    break
            if is_duplicate:
                break
        if not is_duplicate:
            unique_messages.append(line)
            for n in n_values:
                ngrams = generate_ngrams(normalized_text, n)
                seen_ngrams[n].update(ngrams)
    
    return unique_messages

# Funkcja wczytująca dane z pliku tekstowego
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

# Funkcja zapisująca dane do pliku tekstowego
def save_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for line in data:
            file.write(line.lower() + '\n')

if __name__ == "__main__":
    # Utworzenie parsera argumentów
    parser = argparse.ArgumentParser(description='Process SMS data to remove duplicates.')
    parser.add_argument('--input_file', type=str, default='SMSSmishCollection.txt', help='Path to the input file containing SMS data')
    parser.add_argument('--output_file', type=str, default='unique_data.txt', help='Path to save the unique SMS data')

    # Parsowanie argumentów z linii poleceń
    args = parser.parse_args()

    # Jeśli nie podano argumentów, ustaw domyślne ścieżki plików
    if args.input_file == 'SMSSmishCollection.txt':
        current_directory = os.getcwd()
        args.input_file = os.path.join(current_directory, 'SMSSmishCollection.txt')
    
    if args.output_file == 'unique_data.txt':
        current_directory = os.getcwd()
        args.output_file = os.path.join(current_directory, 'unique_data.txt')

    # Wczytanie danych z pliku tekstowego
    messages = load_data(args.input_file)

    # Usuwanie duplikatów krok po kroku
    all_n_values = list(range(4, 11))
    unique_messages = messages
    for n in all_n_values:
        unique_messages = remove_duplicates(unique_messages, [n])

    # Zapisanie unikalnych wiadomości do nowego pliku tekstowego
    save_data(unique_messages, args.output_file)

    print(f"Unikalne wiadomości zostały zapisane do pliku: {args.output_file}")
