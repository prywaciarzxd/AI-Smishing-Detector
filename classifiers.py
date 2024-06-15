import argparse
import csv
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class SmishingClassifier:
    def __init__(self, token_results_file, messages_file, test_size=0.3, random_state=42):
        self.token_results_file = token_results_file
        self.messages_file = messages_file
        self.test_size = test_size
        self.random_state = random_state
        self.features = None  # Inicjalizacja cech jako pusta zmienna
        
    def load_features(self):
        features = {}
        with open(self.token_results_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                features[row['Token']] = int(row['Count'])
        self.features = features  # Przypisanie cech do zmiennej instancji
        return features

    def load_messages(self):
        messages = []
        labels = []
        with open(self.messages_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Pominięcie pierwszego wiersza z nagłówkami
            for row in reader:
                labels.append(int(row[0]))
                messages.append(row[1])
        return messages, labels
    
    def train(self, classifier, epochs=10, batch_size=32):
        if self.features is None:
            self.load_features()  # Załaduj cechy, jeśli nie zostały jeszcze załadowane
        
        messages, labels = self.load_messages()
        messages_train, messages_test, labels_train, labels_test = train_test_split(
            messages, labels, test_size=self.test_size, random_state=self.random_state
        )
        vectorizer = CountVectorizer(vocabulary=self.features.keys())
        X_train = vectorizer.fit_transform(messages_train).toarray()
        X_test = vectorizer.transform(messages_test).toarray()
        self.vectorizer = vectorizer

        if isinstance(classifier, Sequential):
            labels_train = to_categorical(labels_train, num_classes=2)
            labels_test = to_categorical(labels_test, num_classes=2)

        start_time = time.time()
        self.classifier = classifier
        
        if isinstance(classifier, Sequential):
            input_shape = (len(self.features),)  # Ustal input_shape na podstawie liczby cech
            self.classifier.add(Dense(128, input_shape=input_shape, activation='relu'))
            self.classifier.add(Dropout(0.5))
            self.classifier.add(Dense(64, activation='relu'))
            self.classifier.add(Dropout(0.5))
            self.classifier.add(Dense(2, activation='softmax'))
            self.classifier.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
            self.classifier.fit(X_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=0)
        else:
            self.classifier.fit(X_train, labels_train)
        
        self.train_time = time.time() - start_time
        
        self.messages_test = messages_test
        self.labels_test = labels_test
        
    def predict(self):
        X_test = self.vectorizer.transform(self.messages_test).toarray()
        start_time = time.time()
        if isinstance(self.classifier, Sequential):
            predictions = self.classifier.predict(X_test)
            predictions = predictions.argmax(axis=-1)
        else:
            predictions = self.classifier.predict(X_test)
        self.predict_time = time.time() - start_time
        return predictions
    
    def evaluate(self):
        predictions = self.predict()
        accuracy = accuracy_score(self.labels_test, predictions)
        precision = precision_score(self.labels_test, predictions)
        recall = recall_score(self.labels_test, predictions)
        f1 = f1_score(self.labels_test, predictions)
        tn, fp, fn, tp = confusion_matrix(self.labels_test, predictions).ravel()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        return accuracy, precision, recall, f1, fpr, tpr

if __name__ == "__main__":
    # Utworzenie parsera argumentów
    parser = argparse.ArgumentParser(description='Train and evaluate SMS classification models.')
    parser.add_argument('token_results_file', type=str, nargs='?', default='tokeny.csv',
                        help='Path to the token results CSV file (default: "tokeny.csv")')
    parser.add_argument('messages_file', type=str, nargs='?', default='dataset.csv',
                        help='Path to the labeled messages CSV file (default: "dataset.csv")')
    parser.add_argument('--test_size', type=float, default=0.3, help='Size of the test dataset (default: 0.3)')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility (default: 42)')

    # Parsowanie argumentów z linii poleceń
    args = parser.parse_args()

    # Użycie klasyfikatorów
    classifiers = [
        ("Naive Bayes", MultinomialNB(alpha=1)),
        ("SVM", SVC(kernel='rbf', C=0.8)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=3)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=10)),
        ("Random Forest", RandomForestClassifier(n_estimators=50)),
        ("Logistic Regression", LogisticRegression(C=200, penalty='l2', solver='lbfgs', max_iter=100)),
    ]

    results = []

    for name, classifier in classifiers:
        smishing_classifier = SmishingClassifier(args.token_results_file, args.messages_file, args.test_size, args.random_state)
        smishing_classifier.train(classifier)
        
        accuracy, precision, recall, f1, fpr, tpr = smishing_classifier.evaluate()
        results.append((name, accuracy, precision, recall, f1, smishing_classifier.train_time, fpr, tpr))
        print(f"Ewaluacja klasyfikatora {name}:")
        print(f"Dokładność klasyfikacji: {accuracy:.4f}")
        print(f"Precyzja: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"FPR (False Positive Rate): {fpr:.4f}")
        print(f"TPR (True Positive Rate): {tpr:.4f}")   
        print(f"Czas trenowania modelu: {smishing_classifier.train_time:.4f} sekund")
        print(f"Czas przewidywania: {smishing_classifier.predict_time:.4f} sekund")
        print()

    # Sortowanie wyników
    results.sort(key=lambda x: (-x[1], x[5], -x[7]))  # Sortowanie według dokładności (malejąco), czasu trenowania (rosnąco) i TPR (malejąco)

    print("Top 2 klasyfikatory na podstawie dokładności:")
    for result in results[:2]:
        print(f"Klasyfikator: {result[0]}, Dokładność: {result[1]:.4f}, Precyzja: {result[2]:.4f}, Recall: {result[3]:.4f}, F1-Score: {result[4]:.4f}, Czas trenowania: {result[5]:.4f} sekund, FPR: {result[6]:.4f}, TPR: {result[7]:.4f}")
