import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def split_data(data, target_column, test_size=0.2, random_state=42):
    try:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None, None

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        return accuracy, report, matrix
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None, None, None

def main():
    file_path = "data.csv"
    target_column = "target"
    data = load_data(file_path)
    if data is not None:
        X_train, X_test, y_train, y_test = split_data(data, target_column)
        if X_train is not None:
            model = train_model(X_train, y_train)
            if model is not None:
                accuracy, report, matrix = evaluate_model(model, X_test, y_test)
                if accuracy is not None:
                    print(f"Model Accuracy: {accuracy:.3f}")
                    print("Classification Report:")
                    print(report)
                    print("Confusion Matrix:")
                    print(matrix)

if __name__ == "__main__":
    main()