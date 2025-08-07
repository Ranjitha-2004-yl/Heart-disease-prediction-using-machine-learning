from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained RandomForest model
model = joblib.load('random_forest_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        age = float(request.form['age'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        thalach = float(request.form['thalach'])
        oldpeak = float(request.form['oldpeak'])
        sex = int(request.form['sex'])  # Assuming sex is encoded as 1 or 0
        cp = int(request.form['cp'])  # Assuming cp is encoded as 0, 1, 2, or 3
        fbs = int(request.form['fbs'])  # 0 or 1
        restecg = int(request.form['restecg'])  # 0, 1, or 2
        exang = int(request.form['exang'])  # 0 or 1
        slope = int(request.form['slope'])  # 0, 1, or 2
        ca = int(request.form['ca'])  # 0 to 4
        thal = int(request.form['thal'])  # 0, 1, 2, or 3

        # Create a numpy array with input features
        features = np.array([[
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
        ]])
        print(features)

        # Make the prediction
        prediction = model.predict(features)

        # Return prediction
        print(prediction)
        if prediction[0] == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"

        return render_template('index.html', result=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
