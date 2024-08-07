from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('D:/Breast Cancer Prediction/brest_cancer_knn_new.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict(flat=True)
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]
        result = {
            'prediction': 'Malignant (Cancerous)' if prediction[0] == 1 else 'Benign (Not Cancerous)',
            'probability': round(prediction_proba, 2)
        }
        return render_template('result.html', result=result)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
