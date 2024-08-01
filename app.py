from flask import Flask,Request,jsonify,render_template,request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler=pickle.load(open('models/scaler2.pkl','rb'))
ridge=pickle.load(open('models/ridge.pkl','rb'))



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temprature=float(request.form.get('Temprature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        new_scaled_data=scaler.transform([[Temprature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        prediction=ridge.predict(new_scaled_data)
        return render_template('home.html',prediction=prediction[0])
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
