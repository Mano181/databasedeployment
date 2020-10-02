# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:31:53 2020

@author: admin
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import pickle


app = Flask(__name__)
model=pickle.load(open('prediction.pkl','rb'))

#ENV = 'dev'

#if ENV == 'dev':
 #   app.debug=True
  #  app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg2://postgres:1234@localhost/diabetes'
#else:
 #   app.debug = False
  #  app.config['SQLALCHEMY_DATABASE_URI'] = ''
    
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] =  False

#db = SQLAlchemy(app)

#class predictor(db.Model):
 #   __tablename = 'predictor'
  """  Pregnancies = db.Column(db.Integer)
    Glucose = db.Column(db.Integer,primary_key=True)
    BloodPressure = db.Column(db.Integer)
    SkinThickness = db.Column(db.Integer)
    Insulin = db.Column(db.Integer)
    BMI = db.Column(db.Integer)
    DiabetesPedigreeFunction = db.Column(db.Integer)
    Age = db.Column(db.Integer)

    def __init__(self,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
        self.Pregnancies=Pregnancies
        self.Glucose =Glucose
        self.BloodPressure=BloodPressure
        self.SkinThickness =SkinThickness 
        self.Insulin= Insulin
        self.BMI= BMI
        self.DiabetesPedigreeFunction= DiabetesPedigreeFunction
        self.Age= Age
    

     """   
        
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    
    
    -------
    None.

    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction=model.predict(final_features)
    """"if request.method=='POST':
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']
    if db.session.query(predictor).filter(predictor.Glucose>=0):
        data = predictor( Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
        db.session.add(data)
        db.session.commit()"""
    
    output = prediction[0]
    if(output==1):
        return render_template('index.html',prediction_text='sorry you have diabetes')
    else:
        return render_template('index.html',prediction_text='Hurrry you not have diabetes')
    
    #return render_template('index.html',prediction_text='Employee Salary should be $ {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)
