import random
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

#در اینجا با استفاده از کتابخانهی panda فایل های csv رو میخونیم
credits_df = pd.read_csv(r"C:\Users\zzz\Desktop\credits.csv")
movies_df = pd.read_csv(r"C:\Users\zzz\Desktop\movies.csv")
#برای نمایش بهتر دیتاست تنظیمات جدید رو با استفاده از کتابخانهی panda اعمال میکنیم
pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 20)
'''print(credits_df.head())
print("\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
print(movies_df.head())
print("\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
'''
#جوین کردن دو فایل دیتاست بر روی یک ستون مشترک
movies_df = movies_df.merge(credits_df, on = "title")
#در اینجا ستون هایی رو  انتخاب میکنیم که بیشتر به رسیدن ما به جواب کمک میکنند
movies_df = movies_df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew','release_date']]

#print(movies_df.head)

#print(movies_df.isnull().sum())

#خونه هایی که مقداری ندارن رو حذف میکنیم تا جواب دقیق تری بدست آوریم
movies_df.dropna(inplace=True)

#print(movies_df.duplicated().sum())

#با استفادده از کتابخانه یast eaval در قسمت هایی که به تابع میدهیم مقدادیر منتسب به name را جدا میکنیم
def covert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l
#در این قسمت ستونهای مورد نظر رو به تابع convert میدهیم که کارش را انجام دهد
movies_df['genres'] = movies_df['genres'].apply(covert)
movies_df['keywords'] = movies_df['keywords'].apply(covert)
#print(movies_df.head())

#در این قسمت ستونهای مورد نظر رو به تابع convert میدهیم که کارش را انجام دهد
#با گذاشتنcounter تنها از سه اسم اول استفاده میکنیم
def covert3(obj):
    l = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            l.append(i['name'])
            counter += 1
        else:
            break
    return l

movies_df['cast'] = movies_df['cast'].apply(covert3)

#با ساخت یک تابع وبا کمک کتابخانهی مذکور در میان ستون crew به دنبال افرادی میگردیم که
#که شغلشان کارگردان است
def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
    return l

movies_df['crew'] = movies_df['crew'].apply(fetch_director)

#در اینجا با استفاده از lambda فواصل بین کلمات را برمیداریم تا نتیجه ی بدست آمده دقیق تر باشد
movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())
movies_df['genres'] = movies_df['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies_df['cast'] = movies_df['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies_df['crew'] = movies_df['crew'].apply(lambda x:[i.replace(" ", "") for i in x])
movies_df['release_date'] = movies_df['release_date'].apply(lambda x:[i.replace("-"and"/", "") for i in x])
#برای اینکه تعداد ستون ها رو کاهش بدیم و اونهارو سازماندهی کینم،محتویات چند ستون رو در یک ستون میریزیم


crew = str(input("do you want crew to be considerd?(y/n):"))
cast = str(input("do you want cast to be considerd?(y/n):"))
year = str(input("do you want production year to be considerd?(y/n):"))

if ((crew=="y") and (cast=="y") and (year=="y")):
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + \
                        movies_df['crew'] + movies_df['release_date']

elif((crew=="y") and (cast=="n") and (year=="n")):
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] +  movies_df['crew']

elif ((crew == "y") and (cast == "y") and (year == "n")):
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + \
                        movies_df['crew']

elif((crew=="y") and (cast=="n") and (year=="y")):
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords']  + \
                        movies_df['crew'] + movies_df['release_date']

elif((crew=="n") and (cast=="y") and (year=="y")):
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + \
                        + movies_df['release_date']

elif((crew=="n") and (cast=="n") and (year=="y")):
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['release_date']

elif((crew=="n") and (cast=="y") and (year=="n")):
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast']

elif((crew=="n") and (cast=="n") and (year=="n")):
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords']
else:
    print("you didn't enter right input and the recommedation is on default mode.")
    movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + \
                        movies_df['crew'] + movies_df['release_date']

#movies_df['tags'] = movies_df['overview'] + movies_df['genres'] + movies_df['keywords'] + movies_df['cast'] + movies_df['crew'] + movies_df['release_date']
#print(movies_df.head())

#برای اینکه تعداد ستون ها رو کاهش بدیم و اونهارو سازماندهی کینم،محتویات چند ستون رو در یک ستون میریزیم
new_df = movies_df[['movie_id', 'title', 'tags']]
#print(new_df.head())

#براکت ها و ویرگول هارو پاک میکنیم
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
#print(new_df.head())

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
#print(new_df["tags"][0])

#کلمات رو به وکتور و توکن تبدیل میکند
cv = CountVectorizer(max_features=5000, stop_words='english')

#print(cv.fit_transform(new_df['tags']).toarray().shape)

#در اینجا برای دست پیدا کردن به سرعت بالاتر توکن ها رو در آرایه میریزیم
vectors = cv.fit_transform(new_df['tags']).toarray()
#print(vectors[0])

len(cv.get_feature_names_out())

#از کتابخانهی nltk قسمت ریشه گیری رو اضافه میکینم
#در این پروسه کلمات رو به ریشهی خودشون برمیگردونیم تا تجلیل کرنشون برای ما آسون تر بشه
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)


#print(cosine_similarity(vectors))
#print(cosine_similarity(vectors).shape)

similarity = cosine_similarity(vectors)
#print(similarity[0])
#print(similarity[0].shape)


enumerated_list = sorted(list(enumerate(similarity[0])), reverse= True, key= lambda x:x[1])[1:15]
#print(enumerated_list)

def recomend(movie):
        try:
            movie_index = new_df[new_df['title'] == movie].index[0]
            distances = similarity[movie_index]
            movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:15]
            counter = 0
            moviee = []
            for i in movie_list:
                moviee.append([new_df.iloc[i[0]].title])

            for j in range(0, 5):
                print(moviee[random.randint(0, 13)])
        except:
            print("sorry your choosen movie is not avalable in our dataset,pleas try another one.")
while(1):
    x = str(input("movie name: "))
    if (x=="0"):
        break

    recomend(x)
