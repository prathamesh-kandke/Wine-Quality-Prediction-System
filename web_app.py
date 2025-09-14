import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request, url_for
import pickle

app=Flask(__name__)

with open('wine_sc', 'rb') as f:
    sc = pickle.load(f)
with open('wine_model', 'rb') as f:
    model1 = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    a1 = float(request.form['a1'])
    a2 = float(request.form['a2'])
    a3 = float(request.form['a3'])
    a4 = float(request.form['a4'])
    a5 = float(request.form['a5'])
    a6 = float(request.form['a6'])
    a7 = float(request.form['a7'])
    a8 = float(request.form['a8'])
    a9 = float(request.form['a9'])
    a10 = float(request.form['a10'])
    a11 = float(request.form['a11'])
    new=np.array([[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]])
    #print(new)

    a = sc.transform(new)
    pred = model1.predict(a)
    pred=pred[0]
    if pred==1:
        z='Good Quality Wine'
    else:
        z='Average Quality Wine'
    return render_template('predict.html', result=z)

if __name__=='__main__':
    app.run(debug=True)