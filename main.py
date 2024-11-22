from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

main = Flask(__name__)

# Load and preprocess dataset
data = pd.read_excel('data.xlsx')  # Replace this with your file path
symptoms = data['Symptoms']
diseases = data['Disease']
medicines = data['Medicine']

# Convert symptoms into numerical features
vectorizer_symptoms = CountVectorizer()
X_symptoms = vectorizer_symptoms.fit_transform(symptoms).toarray()

# Encode target labels (diseases and medicines)
label_encoder_diseases = LabelEncoder()
y_diseases = label_encoder_diseases.fit_transform(diseases)

label_encoder_medicines = LabelEncoder()
y_medicines = label_encoder_medicines.fit_transform(medicines)

# Split data into training and testing sets for disease prediction
X_train_symptoms, X_test_symptoms, y_train_diseases, y_test_diseases = train_test_split(
    X_symptoms, y_diseases, test_size=0.2, random_state=42
)

# Train models for disease prediction
disease_models = {
    "Naive Bayes": MultinomialNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

trained_disease_models = {}
model_dir = './models'
os.makedirs(model_dir, exist_ok=True)  # Ensure the model directory exists

# Load or train and save disease models
for name, model in disease_models.items():
    model_path = os.path.join(model_dir, f'disease_model_{name}.pkl')
    try:
        trained_disease_models[name] = joblib.load(model_path)
    except FileNotFoundError:
        model.fit(X_train_symptoms, y_train_diseases)
        joblib.dump(model, model_path)
        trained_disease_models[name] = model

# Split data into training and testing sets for medicine recommendation
scaler = StandardScaler()
X_diseases = scaler.fit_transform(
    label_encoder_diseases.transform(diseases).reshape(-1, 1)
)
X_train_diseases, X_test_diseases, y_train_medicines, y_test_medicines = train_test_split(
    X_diseases, y_medicines, test_size=0.2, random_state=42
)

# Train models for medicine recommendation
medicine_models = {
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Artificial Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

trained_medicine_models = {}

# Load or train and save medicine models
for name, model in medicine_models.items():
    model_path = os.path.join(model_dir, f'medicine_model_{name}.pkl')
    try:
        trained_medicine_models[name] = joblib.load(model_path)
    except FileNotFoundError:
        model.fit(X_train_diseases, y_train_medicines)
        joblib.dump(model, model_path)
        trained_medicine_models[name] = model

# Home page route
@main.route('/')
def home():
    return render_template('index.html')

# Disease prediction route
@main.route('/disease', methods=['GET', 'POST'])
def disease_prediction():
    predicted_disease = None
    error = None

    if request.method == 'POST':
        input_symptoms = request.form['symptoms']
        selected_disease_model = request.form['disease_model']

        if not input_symptoms.strip():
            error = "Symptoms cannot be empty."
        else:
            try:
                input_vector = vectorizer_symptoms.transform([input_symptoms]).toarray()
                disease_model = trained_disease_models[selected_disease_model]
                disease_prediction = disease_model.predict(input_vector)
                predicted_disease = label_encoder_diseases.inverse_transform(disease_prediction)[0]
            except ValueError:
                error = "Invalid input: Symptoms not recognized."

    return render_template(
        'disease_prediction.html',
        disease_models=list(disease_models.keys()),
        predicted_disease=predicted_disease,
        error=error
    )

# Medicine suggestion route
@main.route('/medicine', methods=['GET', 'POST'])
def medicine_suggestion():
    recommended_medicine = None
    error = None

    if request.method == 'POST':
        input_disease = request.form['disease']
        selected_medicine_model = request.form['medicine_model']

        if not input_disease.strip():
            error = "Disease cannot be empty."
        else:
            try:
                encoded_disease = label_encoder_diseases.transform([input_disease])
                input_vector = scaler.transform(encoded_disease.reshape(-1, 1))
                medicine_model = trained_medicine_models[selected_medicine_model]
                medicine_prediction = medicine_model.predict(input_vector)
                recommended_medicine = label_encoder_medicines.inverse_transform(medicine_prediction)[0]
            except ValueError:
                error = "Invalid input: Disease not recognized."

    return render_template(
        'medicine_suggestion.html',
        medicine_models=list(medicine_models.keys()),
        recommended_medicine=recommended_medicine,
        error=error
    )

if __name__ == "__main__":
    main.run(debug=True)
