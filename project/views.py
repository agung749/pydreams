
import base64
from matplotlib.figure import Figure
from io import BytesIO
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2
from sklearn import svm 
from sklearn.naive_bayes import MultinomialNB, GaussianNB #naive bayes
from sklearn.model_selection import train_test_split 
import random
from sklearn.metrics import classification_report, confusion_matrix
from nltk.corpus import stopwords
from joblib import dump
import os
import json
from django.templatetags.static import static
from joblib import dump
from joblib import load
data= os.path.join('static', 'kritiksaranfinal.xlsx')
df = pd.read_excel(data)
# change label from categorical to numerik
le = LabelEncoder()
label1 = le.fit_transform(df['label'])
df.drop("label", axis=1, inplace=True)
df["label"] = label1
nol = df.loc[df['label'] == 0]
nol = nol.head(100)
satu = df.loc[df['label'] == 1]
satu = satu.head(100)
dua = df.loc[df['label'] == 2]
dua = dua.head(100)
positive = 0
negative = 0
netral = 0
df = pd.concat([nol, satu, dua])
def casefolding(text):
  text = text.lower()                             
  text = re.sub(r'https?://\S+|www\.\S+', '', text) 
  text = re.sub(r'[-+]?[0-9]+', '', text)        
  text = re.sub(r'[^\w\s]','', text)              
  text = re.sub(r' +', ' ', text)                   
  text = text.strip()
  return text
#Tokenizer
kamus_normalisasi = pd.read_csv('https://raw.githubusercontent.com/ksnugroho/klasifikasi-spam-sms/master/data/key_norm.csv')
def tokenize(text):
  words = text.split()
  words = ' '.join([kamus_normalisasi[kamus_normalisasi['singkat'] == word]['hasil'].values[0] if (kamus_normalisasi['singkat'] == word).any() else word for word in text.split()])
  return words

#Stemming
stemmer_factory = StemmerFactory()
ind_stemmer = stemmer_factory.create_stemmer()
def stemming(komentar):
  komentar = ind_stemmer.stem(komentar)
  return komentar 

# Pipeline NLP
def preprocessing(komentar):
  komentar = casefolding(komentar)
  komentar = stemming(komentar)
  komentar = tokenize(komentar)
  return komentar
df['clean'] = df['teks'].apply(preprocessing)
X, y = df.clean, df.label 
vec_tfidf_uni = TfidfVectorizer(ngram_range=(1,1))
vec_tfidf_uni.fit(X)
#ubah dalam bentuk array
X_unigram_tfidf = vec_tfidf_uni.transform(X).toarray()
#Pakai Unigram
data_unigram_tfidf = pd.DataFrame(X_unigram_tfidf, columns=vec_tfidf_uni.get_feature_names_out())

X_baru = np.array(data_unigram_tfidf)
y_label = np.array(y)

chi2_features = SelectKBest(chi2, k = 200) 
X_kbest_features = chi2_features.fit_transform(X_baru, y_label)

# chi2_features.scores_ adalah nilai chi-square, semakin tinggi nilainya maka semakin baik fiturnya
Data =pd.DataFrame(chi2_features.scores_,columns=['Nilai'])

#Menampilkan fitur beserta nilainya
feature =vec_tfidf_uni.get_feature_names_out()
Data['Fitur'] = feature

#Menampilkan fitur-fitur terpilih berdasarkan  nilai tertinggi yang sudah ditetapkan pada Chi-Square
mask =chi2_features.get_support()
new_feature=[]
for bool,f in zip(mask,feature):
    if bool:
        new_feature.append(f)
    selected_feature=new_feature
data_selected_feature = pd.DataFrame(X_kbest_features,columns=selected_feature)

#Save vectorizer.vocabulary_
#Menyimpan vektor dari vocabulary di atas dalam bentuk pickle (.pkl)
pickle.dump(data_selected_feature,open("selected_feature_tf-idf.pkl","wb"))
X = data_selected_feature.values
y = df.label
x_train , x_test , y_train , y_test = train_test_split(X, y, test_size=0.4 , random_state = 0)

# Training Model
#text_algorithm = GaussianNB() 
text_algorithm = MultinomialNB()
#text_algorithm = SVC()
model = text_algorithm.fit(x_train, y_train)

#prediksi
prediksi = model.predict(x_test)
prediksii = model.predict(x_train)


# save the model to disk
dump(model, filename="model_sentiment_naive.joblib")
def predict(request):
  series = pd.Series(prediksi) 
  positive = series.value_counts()[2]
  netral = series.value_counts()[1]
  negative = series.value_counts()[0]
  if request.method == 'POST' :
      pipeline = load("model_sentiment_naive.joblib")
      data_input = str([request.POST.get('kritik')])
      #input
      data_input = preprocessing(data_input)
      tfidf = TfidfVectorizer
      loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("selected_feature_tf-idf.pkl", "rb"))))
      hasil = pipeline.predict(loaded_vec.fit_transform([data_input]))
      if (hasil==0):
          s ="NEGATIVE"
      elif (hasil==1):
          s ="NETRAL"
      else:
          s ="POSITIVE"
      
      data_input = str([request.POST.get('saran')])
      #input
      data_input = preprocessing(data_input)
      tfidf = TfidfVectorizer
      loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("selected_feature_tf-idf.pkl", "rb"))))
      hasil = pipeline.predict(loaded_vec.fit_transform([data_input]))
      if (hasil==0):
          s2 ="NEGATIVE"
      elif (hasil==1):
          s2 ="NETRAL"
      else:
          s2 ="POSITIVE"
      data4={'kritik':s,'saran':s2, 'negative':negative,
      'positive':positive,
      'netral':netral}
      return render(request, 'halamanklasifikasi.html',data4)
  else:
      return render(request, 'halamanklasifikasi.html')

def visualisasi(request):
  series = pd.Series(prediksi) 
  positive = series.value_counts()[2]
  netral = series.value_counts()[1]
  negative = series.value_counts()[0]

  sizes = [positive, netral, negative]
  plt.figure(figsize=(8,8))
  colors = ['green', 'cyan', 'red']
  mylabels= ["Positive", "Netral", "negative"]
  plt.pie(sizes, labels = mylabels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
  plt.legend()
  plt.title('Sentiment Analysis')
  buf = BytesIO()
  plt.savefig(buf,format='png')
  data = base64.b64encode(buf.getbuffer()).decode("ascii")
  
  foto = {
    'foto':data
  }
  return render(request,'visualisasi.html',foto)
  
def home(request):
  return render(request,'login.html')
def us(request):
  return render(request,'us.html')
