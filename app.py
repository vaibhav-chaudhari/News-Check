import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as metrics
import re
import os






app = Flask(__name__)




def clean_article(article):
    art = re.sub("[^A-Za-z0-9' ]", '', str(article))
    art2 = re.sub("[( ' )(' )( ')]", ' ', str(art))
    art3 = re.sub("\s[A-Za-z]\s", ' ', str(art2))
    return art3.lower()


model = joblib.load(open('model.pkl', 'rb'))
cv = joblib.load(open('cv.pkl', 'rb'))
tfidfv = joblib.load(open('tfidfv.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    comment = request.form['news']
    list_comment = [comment]

    list_comment = clean_article(list_comment)
    list_comment = [list_comment]
        

    
  
    
    prediction = model.predict(tfidfv.transform(list_comment))
    output = prediction[0]
  




    
    
     

 
    

    return render_template('index.html', prediction_text='The news is more likely to be as a {} news'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
