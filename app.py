from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('most_imp_eight_attributes_fuel_model.sav', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():

    # takeoff_ground_run = request.form.get('Takeoff ground run')
    # all_eng_service_ceiling = request.form.get('All eng service ceiling')
    # height = request.form.get('Height ft/in')
    # empty_weight = request.form.get('Empty weight lbs')
    # length = request.form.get('Length ft/in')
    # range_nm = request.form.get('Range N.M.')
    # wing_span = request.form.get('Wing span ft/in')
    # gross_weight = request.form.get('Gross weight lbs')

    # height = request.form.get('height')
    # takeoff_ground_run = request.form.get('takeoffGroundRun')
    # all_eng_service_ceiling = request.form.get('allEngServiceCeiling')
    # empty_weight = request.form.get('emptyWeight')
    # length = request.form.get('length')
    # range_nm = request.form.get('range')
    # wing_span = request.form.get('wingSpan')
    # gross_weight = request.form.get('grossWeight')

    # input_query = np.array([[1005.0, 13000.0, 46, 61, 5, 71, 206, 85]])
    # input_query = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

    # result = model.predict()[0]
    # result = model.predict([[1005.0, 13000.0, 46, 61, 5, 71, 206, 85 ,0,0,0,0,0,0,0,0,0]])[0]
    # result = model.predict([[1005.0, 12300.0, 602, 16, 147, 57, 116, 37 ,0,0,0,0,0,0,0,0,0]])[0]
    # result = model.predict([[1005.0, 16000.0, 597, 23, 146, 43, 131, 31 ,0,0,0,0,0,0,0,0,0]])[0]
    # result = model.predict([[1005.0, 13000.0, 592, 23, 146, 43, 131, 31 ,0,0,0,0,0,0,0,0,0]])[0]

    a1 = request.form.get('a1')
    a2 = request.form.get('a2')
    a3 = request.form.get('a3')
    a4 = request.form.get('a4')
    a5 = request.form.get('a5')
    a6 = request.form.get('a6')
    a7 = request.form.get('a7')
    a8 = request.form.get('a8')



    # result = {'a1':a1, 'a2':a2, 'a3':a3, 'a4':a4, 'a5':a5, 'a6':a6, 'a7':a7, 'a8':a8}

    input_query = np.array([[a1,a2,a3,a4,a5,a6,a7,a8, 0,0,0,0,0,0,0,0,0]])
    result = model.predict(input_query)[0]


    return jsonify({'optimum fuel ': str(result)})





    # result = model.predict([[1005.0, 12500.0, 574, 23, 147, 43, 100, 23  ,0,0,0,0,0,0,0,0,0]])[0]


    # return jsonify({'opfuel': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
