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

app = Flask(__name__)

#Text Cleaning
class PreProcessTweets:
    def __init__(self):
        self.stop = set(stopwords.words('english'))
        self.special = string.punctuation
        
    def _removestopwords(self, text):
            if text is not None:
                tokens = [x for x in word_tokenize(text) if x not in self.stop]
                return " ".join(tokens)
            else:
                return None
        
    def _removeURL(self, text):
            return re.sub(r'http\S+', '', text)
    
    def _removehtml(self, text):
        html=re.compile(r'<.*?>')
        return html.sub(r'',text)

    def _removeemoji(self, text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def _removepunct(self, text):
        table=str.maketrans('','',self.special)
        return text.translate(table)

    def _removenum(self, text):
        return re.sub(r'\b[0-9]+\b', '', text)

    def _removesmallwords(self, text):
        shortword = re.compile(r'\W*\b\w{1,2}\b')
        return shortword.sub('', text)

    def clean_tweet(self, tweet):
        tweet=tweet.lower()
        tweet=self._removeURL(tweet)
        tweet=self._removehtml(tweet)
        tweet=self._removeemoji(tweet)
        tweet=self._removepunct(tweet)
        tweet=self._removestopwords(tweet)
        tweet=self._removenum(tweet)
        tweet=self._removesmallwords(tweet)
        tweet=tweet.replace('\s+', ' ')
        return tweet

    def clean_keywords(self, keyword):
        keyword=keyword.lower()
        keyword=self._removeemoji(keyword)
        keyword=keyword.replace('%20', ' ')
        keyword=keyword.replace('\s+', ' ')
        return keyword

#feature creation
class CreateFeature:
    def __init__(self):
        self.Preprocessor=PreProcessTweets()
        self.ps = PorterStemmer()
    
    def avg_word_len(self, tweet):
        tweet=self.Preprocessor.clean_tweet(tweet)
        return np.average([len(i) for i in tweet.split()])

    def num_of_links(self, tweet):
        return len(re.findall(r"http", tweet))

    def kw_weight(self, keyword):
        keyword=self.Preprocessor.clean_keywords(keyword)
        key_dict=pickle.load(open("keyword_dict.pkl", 'rb'))
        stem_key=self.ps.stem(keyword.strip())
        return key_dict[stem_key] if stem_key in list(key_dict.keys()) else 0

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
      
    loaded_model = pickle.load(open("base_model3.pkl", 'rb'))
    cf=CreateFeature()

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