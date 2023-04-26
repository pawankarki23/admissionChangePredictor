import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

## Load the scaler and model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    # convert into np array
    np_array_2d = np.array(list(data.values())).reshape(1,-1)
    print(np_array_2d)

    new_data=scalar.transform(np_array_2d)
    output=regmodel.predict(new_data)
    print(output[0])
    
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    np_array_2d = np.array(data).reshape(1,-1)
    final_input=scalar.transform(np_array_2d)
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="Chance of Admitiion is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     
