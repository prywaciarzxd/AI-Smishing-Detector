import argparse
import re
from collections import defaultdict
import csv
import os

# Funkcja normalizująca tekst
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\d', 'N', text)  # Zamiana cyfr na 'N'
    text = re.sub(r'\W+', ' ', text)  # Zamiana znaków specjalnych na białe spacje
    return text.strip()

# Funkcja tokenizująca tekst i zliczająca tokeny
def tokenize_text(file_path):
    # Inicjalizujemy słowniki defaultdict do zliczania wystąpień tokenów dla etykiet 'ham' i 'smish'
    ham_token_counts = defaultdict(int)
    smish_token_counts = defaultdict(int)
    total_tokens_ham = 0
    total_tokens_smish = 0
    
    # Otwieramy plik do odczytu
    with open(file_path, 'r') as file:
        # Iterujemy przez każdą linię w pliku
        for line in file:
            # Rozdzielamy linię na etykietę i tekst, oddzielone tabulatorem
            label, text = line.strip().split('\t', 1)
            # Tokenizujemy tekst i zamieniamy na małe litery
            tokens = text.lower().split()
            # Zliczamy wystąpienia tokenów dla odpowiedniej etykiety
            if label == 'ham':
                total_tokens_ham += len(tokens)
                for token in tokens:
                    ham_token_counts[token] += 1
            elif label == 'smish':
                total_tokens_smish += len(tokens)
                for token in tokens:
                    smish_token_counts[token] += 1
    
    # Sortujemy tokeny według liczby ich wystąpień dla etykiety 'ham'
    sorted_ham_token_counts = sorted(ham_token_counts.items(), key=lambda x: x[1], reverse=True)
    # Sortujemy tokeny według liczby ich wystąpień dla etykiety 'smish'
    sorted_smish_token_counts = sorted(smish_token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Zwracamy posortowane tokeny dla obu etykiet oraz liczbę wszystkich tokenów
    return sorted_ham_token_counts[:20], sorted_smish_token_counts[:20], total_tokens_ham, total_tokens_smish

# Funkcja zapisująca wyniki do pliku CSV
def save_results_to_csv(sorted_ham_token_counts, sorted_smish_token_counts, output_file_path, total_tokens_ham, total_tokens_smish):
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Token", "Count"])

        # Zapisujemy tokeny dla 'ham' z etykietą 0
        for token, count in sorted_ham_token_counts:
            writer.writerow([token, count])

        # Zapisujemy tokeny dla 'smish' z etykietą 1
        for token, count in sorted_smish_token_counts:
            writer.writerow([token, count])

    return total_tokens_ham + total_tokens_smish

if __name__ == "__main__":
    # Utworzenie parsera argumentów
    parser = argparse.ArgumentParser(description='Tokenize SMS data and save token counts to CSV.')
    parser.add_argument('--input_file', type=str, default='unique_data.txt', help='Path to the input file containing SMS data')
    parser.add_argument('--output_file', type=str, default='tokeny.csv', help='Path to save the unique SMS data')

    # Parsowanie argumentów z linii poleceń
    args = parser.parse_args()

    # Jeśli nie podano argumentów, ustaw domyślne ścieżki plików
    if args.input_file == 'unique_data.txt':
        current_directory = os.getcwd()
        args.input_file = os.path.join(current_directory, 'unique_data.txt')
    
    if args.output_file == 'tokeny.csv':
        current_directory = os.getcwd()
        args.output_file = os.path.join(current_directory, 'tokeny.csv')

    # Tokenizacja tekstu i zapis wyników do pliku CSV
    sorted_ham_token_counts, sorted_smish_token_counts, total_tokens_ham, total_tokens_smish = tokenize_text(args.input_file)
    total_tokens = save_results_to_csv(sorted_ham_token_counts, sorted_smish_token_counts, args.output_file.replace('.txt', '_token_counts.csv'), total_tokens_ham, total_tokens_smish)

    print(f"Wyniki tokenizacji zostały zapisane do pliku: {args.output_file.replace('.txt', '_token_counts.csv')}")
    print(f"Liczba całkowita wszystkich tokenów smish: {total_tokens_smish}")
    print(f"Liczba całkowita wszystkich tokenów ham: {total_tokens_ham}")
    print(f"Liczba całkowita wszystkich tokenów: {total_tokens}")
