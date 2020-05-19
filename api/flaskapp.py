from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import pymongo
import os
import numpy as np
import joblib
#from sklearn.externals import joblib

loaded_model=joblib.load("./pkl_objects/model.pkl")
loaded_stop=joblib.load("./pkl_objects/stopwords.pkl")
loaded_vec=joblib.load("./pkl_objects/vectorizer.pkl")



app = Flask(__name__)

def classify(document):
    label = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'guilt', 4 :'joy', 5: 'sadness', 6:'shame'}
    X = loaded_vec.transform([document])
    y = loaded_model.predict(X)[0]
    proba = np.max(loaded_model.predict_proba(X))
    #count += 1
    return label[y], proba



class SentimentForm(Form):
    sentiment = TextAreaField('',[validators.DataRequired(),validators.length(min=15)])

@app.route('/')
def index():
    form = SentimentForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    lis = list()
    form = SentimentForm(request.form)
    if request.method == 'POST' and form.validate():
        mydict = { "name": "John", "address": "Highway 37" }

        review = request.form['sentiment']
      
        
        y, proba = classify(review)
        myclient = pymongo.MongoClient("mongodb://localhost:27017")
        mydb = myclient["mydatabase"]
        mycol = mydb["prediction"]
        predictions = {"sentence": review, "predicted class": y, "probability": proba}
        mycol.insert_one(predictions)
        return render_template('results.html',content=review,prediction=y,probability=round(proba*100, 2) , database = [i for i in mycol.find()], len = len([i for i in mycol.find()]))
    return render_template('reviewform.html', form=form)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
