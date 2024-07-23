#flask,scikit-learn,pandas,pickle-mixin
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import csv

app = Flask(__name__)
data = pd.read_csv('gemstone.csv')

with open('LassoModelDiamond.pkl', 'rb') as file:
    pipe = pickle.load(file)

@app.route("/")
def index():

    cut = sorted(data['cut'].unique())
    color = sorted(data['color'].unique())
    clarity = sorted(data['clarity'].unique())
    return render_template('index.html',cut = cut, clarity=clarity, color=color)


@app.route("/predict", methods = ['POST'])
def predict():

    carat = request.form.get('carat')
    cut = request.form.get('cut')
    color = request.form.get('color')
    clarity = request.form.get('clarity')
    depth = request.form.get('depth')
    table = request.form.get('table')
    x = request.form.get('x')
    y = request.form.get('y')
    z = request.form.get('z')



    print(carat,cut,color,clarity,depth,table,x,y,z)
    input = pd.DataFrame([[carat,cut,color,clarity,depth,table,x,y,z]],columns = ['carat','cut','color','clarity','depth','table','x','y','z'])
    prediction = pipe.predict(input)[0]


    return str(np.round(prediction)*10)


if __name__=="__main__":
    app.run(debug=True)