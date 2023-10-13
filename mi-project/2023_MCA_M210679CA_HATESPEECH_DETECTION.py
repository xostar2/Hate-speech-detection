'''
TITLE - HATESPEECH DETECTION
DEFINATION OF PROBLEM- Here we want to detected the hate speeches and categories in two class that
is "HATE SPEECH" or "NOT A HATE SPEECH"

MEMBERS-RAJ KISHAN
ROLL NO-M210678CA
EMAIL-raj_m210679ca@nitc.ac.in
SUBMITSION DATE- 14 - 4 - 2023


'''
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#Stop words are common words like ‘the’, ‘and’, ‘I’, etc. that are very frequent in text, and so don’t
# convey insights into the specific topic of a document.
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
from nltk.util import pr
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words("english"))



tweet_df = pd.read_csv('train.csv')
print(tweet_df.head())


print("========================================================================")
tweet_df.info()
print("========================================================================")




# printing random tweets 
print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")


print("========================================================================")

#creating a function to process the data
def data_processing(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"https\S+|www\S+http\S+", '', tweet, flags = re.MULTILINE)
    tweet = re.sub(r'\@w+|\#','', tweet)
    tweet = re.sub(r'[^\w\s]','',tweet)
    tweet = re.sub(r'ð','',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_tweets = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_tweets)



#here we appling the data cleaning process
tweet_df.tweet = tweet_df['tweet'].apply(data_processing)

#dropping all the duplicate tweets
tweet_df = tweet_df.drop_duplicates('tweet')

# Lemmatization is another technique used to reduce inflected words to their
# root word. It describes the algorithmic process of identifying an 
# inflected word's “lemma” (dictionary form) based on its intended meaning.

lemmatizer = WordNetLemmatizer()
def lemmatizing(data):
    tweet = [lemmatizer.lemmatize(word) for word in data]
    return data

# printing the data to see the effect of preprocessing
print(tweet_df['tweet'].iloc[0],"\n")
print(tweet_df['tweet'].iloc[1],"\n")
print(tweet_df['tweet'].iloc[2],"\n")
print(tweet_df['tweet'].iloc[3],"\n")
print(tweet_df['tweet'].iloc[4],"\n")

tweet_df.info()
print("========================================================================")

tweet_df['label'].value_counts()



#bar daigram of dataset distribution
fig = plt.figure(figsize=(5,5))
sns.countplot(x='label', data = tweet_df)


#paichart of data distribution

fig = plt.figure(figsize=(7,7))
colors = ("red", "gold")
wp = {'linewidth':2, 'edgecolor':"white"}
tags = tweet_df['label'].value_counts()
explode = (0.1, 0.1)
tags.plot(kind='pie',autopct = '%1.1f%%', shadow=True, colors = colors, startangle =90, 
         wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')
print("========================================================================")
non_hate_tweets = tweet_df[tweet_df.label == 0]
non_hate_tweets.head()

#here we showing most frequent word in non hate tweet show visually

text = ' '.join([word for word in non_hate_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='white')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in non hate tweets', fontsize = 19)
plt.show()
print("========================================================================")

#show all the data with the lable one
neg_tweets = tweet_df[tweet_df.label == 1]
neg_tweets.head()


#here we showing most frequent word in  hate tweet show visually
text = ' '.join([word for word in neg_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most frequent words in hate tweets', fontsize = 19)
plt.show()
print("========================================================================")

vect = TfidfVectorizer(ngram_range=(1,2)).fit(tweet_df['tweet'])

feature_names = vect.get_feature_names()
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features: \n{}".format(feature_names[:20]))


vect = TfidfVectorizer(ngram_range=(1,3)).fit(tweet_df['tweet'])

feature_names = vect.get_feature_names()
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features: \n{}".format(feature_names[:20]))
print("========================================================================")
#model BUilding
X = tweet_df['tweet']
Y = tweet_df['label']
X = vect.transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test: ", (x_test.shape))
print("Size of y_test: ", (y_test.shape))

#using logical reggresion here and try to fint fir of data and how accurent our prediction is
 
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_predict = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_predict, y_test)
print("Test accuarcy: {:.2f}%".format(logreg_acc*100))


print(confusion_matrix(y_test, logreg_predict))
print("\n")
print(classification_report(y_test, logreg_predict))
print("========================================================================")
#we are ploting of cunfusion metrix of our ml model
# It is a table that is used in classification problems to assess where errors in the
# model were made.
# The rows represent the actual classes the outcomes should have been. While the
# columns represent the predictionswe have made. Using this table it is easy to 
# see which predictions are wrong.


style.use('classic')
cm = confusion_matrix(y_test, logreg_predict, labels=logreg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
disp.plot()

print("========================================================================")
# What is GridSearchCV in Python?
#GridSearchCV is a technique for finding the optimal parameter values from a given set 
#of parameters in a grid. It's essentially a cross-validation technique. The model as
# well as the parameters must be entered. After extracting the best parameter values,
# predictions are made.

from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
param_grid = {'C':[100, 10, 1.0, 0.1, 0.01], 'solver' :['newton-cg', 'lbfgs','liblinear']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv = 5)
grid.fit(x_train, y_train)
print("Best Cross validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)

y_pred = grid.predict(x_test)
print("========================================================================")

x=np.array(tweet_df["tweet"])
y=np.array(tweet_df["label"])

cv=CountVectorizer()
x=cv.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.33 ,random_state= 42)
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)

test_data=input("tweet:")

df=cv.transform([test_data]).toarray()
if clf.predict(df)==1:
    print("Hate speech")
else :
    print("Not a Hate Speech")



DecisionTreeClassifier