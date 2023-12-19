import pickle
from flask import Flask, render_template, request
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask_cors import CORS
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

with open('attrition_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
app_flask = Flask(__name__)
CORS(app_flask)

@app_flask.route('/')
def index():
    return render_template('index1.html')

@app_flask.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form.get('age'))
        travelclass = int(request.form.get('travelclass'))
        department = int(request.form.get('department'))
        distance = float(request.form.get('distance'))
        edulvl = int(request.form.get('edulvl'))
        edufield = int(request.form.get('edufield'))
        envsatis = int(request.form.get('envsatis'))
        gender = int(request.form.get('gender'))
        jobinvolvement = int(request.form.get('jobinvolvement'))
        joblvl = int(request.form.get('joblvl'))
        jobrole = int(request.form.get('jobrole'))
        jobsatis = int(request.form.get('jobsatis'))
        maritalsatus = int(request.form.get('maritalsatus'))
        nocompanies = int(request.form.get('nocompanies'))
        overtime = int(request.form.get('overtime'))
        performancerat = int(request.form.get('performancerat'))
        relationshiplvl = int(request.form.get('relationshiplvl'))
        stockoption = int(request.form.get('stockoption'))
        experience = int(request.form.get('experience'))
        trainingtime = int(request.form.get('trainingtime'))
        worklifebal = int(request.form.get('worklifebal'))
        yearsworked = int(request.form.get('yearsworked'))
        yrsincurrole = int(request.form.get('yrsincurrole'))
        yrsperformance = int(request.form.get('yrsperformance'))
        yrscurmanager = int(request.form.get('yrscurmanager'))
        salary = int(request.form.get('salary'))
        percenthike = int(request.form.get('percenthike'))
        # Process the form data and make predictions
        
        input_data = [[travelclass,department,distance,edulvl,edufield,envsatis,gender,jobinvolvement,joblvl,jobrole,jobsatis,maritalsatus,salary,nocompanies,overtime,percenthike,performancerat,relationshiplvl,stockoption,experience,trainingtime,worklifebal,yearsworked,yrsincurrole,yrsperformance,yrscurmanager,age]]
        prediction = model.predict(input_data)
        
        decision_values = model.decision_function(input_data)

# Extract probabilities for leave and stay
        probability_leave = 1 / (1 + np.exp(-decision_values))
        probability_stay = 1 - probability_leave
        
        features=["BusinessTravel","Department","DistanceFromHome","Education","EducationField","EnvironmentSatisfaction","Gender","JobInvolvement","JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","NumCompaniesWorked","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","Age"]
        max_feature = features[np.argmax(np.abs(decision_values))]
        max_contri = np.max(np.abs(decision_values))

        # Render the prediction result on the template
        if (prediction[0]==0.0):
            text = f"The Employee Will Stay. Probability of Staying: {float(probability_stay[0]):.2%} | \t Factor Contributing -  {max_feature}:\n{round(float(max_contri * 100), 2)}%"
        else:
            text = f"The Employee Will Leave. Probability of Leaving: {float(probability_leave[0]):.2%} | \t Factor Contributing - {max_feature}\n{round(float(max_contri * 100), 2)}%"

        # Render the prediction result on the template
        return render_template('index1.html', prediction_text=text)
    
@app_flask.route('/analysis1')
def analysis1():
    return render_template('analysis1.html')
if __name__ == '__main__':
    app_flask.run(debug=True)
