from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
import pickle
from nltk.stem import WordNetLemmatizer
from naive_bayes_v2 import naive_bayes



english_stemmer = SnowballStemmer("english")

analyzer = CountVectorizer().build_analyzer()


app = Flask(__name__)

count_file_name = 'count.pkl'
tfidf_transformer_file_name = 'tfidf.pkl'
tfidf_train_file_name = 'trained_tfidf.pkl'


movies_file_name = 'movies_title.pkl'

classification_X_file = 'x_train.pkl'
classification_Y_file = 'y_train.pkl'

x_file = open(classification_X_file,'rb')
y_file = open(classification_Y_file,'rb')

X_train = pickle.load(x_file)
Y_train = pickle.load(y_file)

genres =  np.array(['Comedy', 'Action','Animation', 'Romance', 'Adventure', 'Horror'])

classifier = naive_bayes()
classifier.initialize(X_train, Y_train, list(genres))

MEAN_MULTIPLIER  = 1.19

lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')

def stemming(text):
    return (english_stemmer.stem(w) for w in analyzer(text))

count_vector = joblib.load(count_file_name)
tfidf_transformer = joblib.load(tfidf_transformer_file_name)
df_movies = pd.read_pickle(movies_file_name)
tfidf_trained = joblib.load(tfidf_train_file_name)

def process_test_classification(text):
    # print(type(text))   
    text = re.sub('[^a-z\s]', '', text.lower())
    text = [lemmatizer.lemmatize(w) for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)

def process_text(text):
    text = re.sub('[^a-z\s]','', text.lower())
    text = [w for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)

def predict(text):
    text = process_text(text)
    pred = classifier.predict(text.split(' '))
    print(pred)
    lst = []
    for i in range(genres.shape[0]):
        lst.append(pred[genres[i]])
    lst = np.array(lst)
    ids = lst > (lst.mean()*MEAN_MULTIPLIER)
    lst = lst[ids]
    pred_genre = genres[ids]
    arg = np.argsort(lst)
    return {x:y for x,y in zip(pred_genre[arg],lst[arg])}

@app.route('/search', methods=['POST','GET'])
def search():
    try:
        query = request.form['query']
        query = process_test_classification(query)
        query_matrix = count_vector.transform([query])
        query_tfidf = tfidf_transformer.transform(query_matrix)
        sim_score = cosine_similarity(query_tfidf, tfidf_trained)
        sorted_indexes = np.argsort(sim_score).tolist()[0]
        top_results = sorted_indexes[-20:]
        top_results.reverse()
        movies_list = df_movies.iloc[top_results]
    except Exception as e:
        print("Error:",e)
        return render_template('search_v1.html')
        print('elseeee')
    return render_template('search_result_v1.html', list = movies_list)

@app.route('/classify', methods=['POST','GET'])
def classify():
    try:
        query = request.form['query']
        result = predict(query)
    except Exception as e:
        print("Error:",e)
        return render_template('classify.html')
    return render_template('classify_result.html', text_area_value = query,result = result)

@app.route('/')
def index():
    return render_template('search_v1.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
