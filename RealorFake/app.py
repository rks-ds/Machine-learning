from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import numpy as np
import re
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop=set(stopwords.words('english'))
from package.Clean_data import PreProcessTweets


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
      
    loaded_model = pickle.load(open("base_model5.pkl", 'rb'))
    cf=PreProcessTweets()

    if request.method == 'POST':
        message = request.form['message']
        keyword = request.form['keyword']
        if message=='':
            my_prediction=np.array([0])
        else:
            df=pd.DataFrame.from_dict({'avg_word_len':[cf.avg_word_len(message)],
                                       'numoflinks':[cf.num_of_links(message)],'keyword_weight':[cf.kw_weight(keyword)]})
            my_prediction = loaded_model.predict(df)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)