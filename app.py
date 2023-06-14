from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('ml_model_23_june.sav', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    #   a1 = EmptyWeight
    #   a2 = GrossWeight
    #   a3 = Length
    #   a4 = Height
    #   a5 = WingSpan
    #   a6 = Range
    #   a7 = GroundRun
    #   a8 = ServiceCeiling

    a1 = request.form.get('a1')
    a2 = request.form.get('a2')
    a3 = request.form.get('a3')
    a4 = request.form.get('a4')
    a5 = request.form.get('a5')
    a6 = request.form.get('a6')
    a7 = request.form.get('a7')
    a8 = request.form.get('a8')

    input_query = np.array([[a1, a2, a3, a4, a5, a6, a7, a8]])
    result = model.predict(input_query)[0]

    return jsonify({'optimum fuel ': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
