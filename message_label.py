import argparse
import csv
import os
import subprocess

if __name__ == "__main__":
    # Utworzenie parsera argumentów
    parser = argparse.ArgumentParser(description='Convert SMS data file to CSV with labeled messages.')
    parser.add_argument('--input_file', type=str, default='unique_data.txt', help='Path to the input file containing SMS data')
    parser.add_argument('--output_file', type=str, default='dataset.csv', help='Path to save the CSV file')

    # Parsowanie argumentów z linii poleceń
    args = parser.parse_args()

    # Jeśli ścieżka do pliku wejściowego nie jest podana, użyj domyślnej ścieżki
    if args.input_file == 'unique_data.txt':
        current_directory = os.getcwd()
        args.input_file = os.path.join(current_directory, 'unique_data.txt')
    
    # Jeśli ścieżka do pliku wyjściowego nie jest podana, użyj domyślnej ścieżki
    if args.output_file == 'dataset.csv':
        current_directory = os.getcwd()
        args.output_file = os.path.join(current_directory, 'dataset.csv')

    # Otwarcie pliku wejściowego i zapisanie do pliku CSV z etykietami
    with open(args.input_file, 'r') as infile, open(args.output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Zapis nagłówka
        writer.writerow(['label', 'message'])
        
        # Iteracja przez każdą linię w pliku tekstowym
        for line in infile:
            # Podzielenie linii na etykietę i treść wiadomości po pierwszym wystąpieniu tabulacji
            label, message = line.strip().split('\t', 1)
            # Zamiana etykiety 'ham' na 0 i 'smish' na 1
            label = 0 if label == 'ham' else 1
            # Zapis do pliku CSV
            writer.writerow([label, message])

    print("Konwersja zakończona pomyślnie.")

