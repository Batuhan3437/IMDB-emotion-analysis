import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

#delimiter tab'la ayrıldığı için kullanıyoruz.Quotings=3 de noktalama işareti yüzünden kullanıldı
df=pd.read_csv("reading_data/NLPlabeledData.tsv",delimiter="\t",quoting=3)

nltk.download("stopwords")

#ilk yorumu göstermek için
sample_review=df.review[0]
#print(sample_review)

#br gibi html taglerinden kurtulmak için beatiful soup kullanıyoruz.
sample_review=BeautifulSoup(sample_review).get_text()
#print(sample_review)

#şimdi noktalama işaretleri ve sayılardan metni temizliyoruz.
sample_review=re.sub("[^a-zA-Z]",' ',sample_review)
#print(sample_review)

#hepsini küçük harf yapmak içşn
sample_review=sample_review.lower()
#print(sample_review)

#ayırmak için split metodunu kullanıyoruz(listeye dönüştü)
sample_review=sample_review.split()
#print(sample_review)


#len(sample_review)
#437 çıktı yani 437 kelimeye ayırmış

#the is are gibi stop wordleri temizleme işlemi w=word
swords=set(stopwords.words("english"))
sample_review=[w for w in sample_review if w not in swords]
#print(sample_review)
#len(sample_review)
#219 kelimeye düştü

#yukarıdaki adımları içeren bir fonksiyon
def process(review):
    review = BeautifulSoup(review).get_text()
    review = re.sub("[^a-zA-Z]", ' ', review)
    review = review.lower()
    review = review.split()
    swords = set(stopwords.words("english"))
    review = [w for w in review if w not in swords]
    return " ".join(review)
#her 1000 review sonrası bir satır yazdırarak review işleminin durumu gözüküyor.
train_x_tum=[]
for r in range (len(df["review"])):
    if (r+1)%1000==0:
        print("No of review processed=",r+1)
    train_x_tum.append(process(df["review"][r]))
    
x=train_x_tum
y=np.array(df["sentiment"])

#train ve test işlemlerini oluşturuyoruz ve ayırıyoruz.
train_x,test_x,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)

#bag of words işlemini yapıyoruz.Kelimeleri 0 ve 1 lere çevirip yapay zekanın anlayacığı şekile getiriyoruz.
#onun için sklearn içinde CountVectorizer ı kullanıyoruz.

vectorizer=CountVectorizer(max_features=5000)
#train verilerimizi feature vectorize matrisine çeviriyoruz.
train_x=vectorizer.fit_transform(train_x)
#train x i dizi yapıyoruz fit ederken çünkü array istiyor.
train_x=train_x.toarray()

train_y=y_train

#print(train_x.shape,train_y.shape)
#çıktı (22500,5000) (22500,)

#fit(modelin eğitilme kısmı) işlemi burda yapılıyor.
model=RandomForestClassifier(n_estimators=100)
model.fit(train_x,train_y)



yorum = str(input("Film hakkındaki yorumunuz nedir: "))

# Yorumu işleyin
yorum = process(yorum)

# Eğitilmiş CountVectorizer'ı yükleyin
vectorizer = CountVectorizer(max_features=5000)

# Kullanıcı girdisini öznitelik vektörüne dönüştürün
yorum = [yorum]
yorum = vectorizer.transform(yorum)
yorum = yorum.toarray()


# Tahmin yapın
test_predict = model.predict(yorum)

# Türkçe sonucu yazdırın
if test_predict[0] == 1:
    print("Tahmin Sonucu: Olumlu")
else:
    print("Tahmin Sonucu: Olumsuz")
#hata alıyorum hata:AttributeError: 'numpy.ndarray' object has no attribute 'lower'

#dogruluk=roc_auc_score(y_test,test_predict)

#print("Doğruluk oranı: %",dogruluk*100)
