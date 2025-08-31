import pickle 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)
app = application 

# import ridge regressor and standard scaler
ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('model/scaler.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        # collect values from form in the same order as training
        data = [float(x) for x in request.form.values()]
        
        # scale input
        final_data = standard_scaler.transform([data])
        
        # make prediction
        output = ridge_model.predict(final_data)[0]
        
        return render_template("home.html", result=output)
    else:
        return render_template("home.html")



if __name__ == "__main__":
    app.run(host="0.0.0.0")
