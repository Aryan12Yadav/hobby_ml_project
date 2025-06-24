from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(x) for x in request.form.values()]
        transformed_data = scaler.transform(np.array(input_data).reshape(1, -1))
        prediction = model.predict(transformed_data)[0]
        return render_template('home.html', prediction_text=f"Predicted House Price: ${prediction:.2f}")
    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)['data']
        input_data = np.array(list(data.values())).reshape(1, -1)
        transformed = scaler.transform(input_data)
        prediction = model.predict(transformed)[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
