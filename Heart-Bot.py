

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model, tree, ensemble
from sklearn.model_selection import train_test_split
from sklearn import metrics

# from google.colab import drive
# drive.mount('/content/gdrive')



df=pd.read_csv("/content/heart.csv")
df.head(10)
df.info() #data analysis

# df.isna().sum()
#print("Since there are no null values and duplicates, our data is good to go for analysis and visualizations")

correlation=df.corr()
pd.DataFrame(correlation['target']).sort_values(by='target',ascending=False)

#making correlation matrix to Visulaizing the data features to find the correlation between them which will infer the important feature
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),linewidth=.01,annot=True,cmap="winter")
plt.show()
plt.savefig('correlationfigure')

df.hist(figsize=(13,15))
plt.savefig('featuresplot')

#Now, we shall find relationship between target and mostly risk factors/features which cause cardiovascular disease.


fig,ax=plt.subplots(figsize=(12,6))
sns.barplot(y='target',x='cp',hue='cp',ax=ax,data=df)
plt.title('Chest Pain vs Heart Disease',size=25)

sns.countplot(data=df,x='sex',hue='target',palette='gist_rainbow')
plt.title('Gender vs target')

pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Age vs Heart Disease Frequency',size=25)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# feature selection
from sklearn.feature_selection import chi2
X = df.drop('target',axis=1)
y = df['target']
chi_scores = chi2(X,y)
chi_scores

p_values = pd.Series(chi_scores[1],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
p_values.plot.bar()

print("The resulting bar plot provides a visual representation of the significance of each feature in the dataset with respect to the target variable. Features with lower p-values are considered more statistically significant and may be more relevant for the classification task. Therefore, this plot can help you make informed decisions about which features to include or exclude from your machine learning model.")

# from sklearn.feature_selection import SelectKBest
# chi2_selector = SelectKBest(chi2, k=2)
# X_kbest = chi2_selector.fit_transform(X, y)

# print('Original number of features:', X.shape)
# print('Reduced number of features:', X_kbest.shape)
# #print(X_kbest)
# # Get the names of the k-best features
# selected_feature_indices = chi2_selector.get_support()
# selected_feature_names = X.columns[selected_feature_indices]

# # Print the names of the selected features
# print("Selected feature names:", selected_feature_names)

df1 = df.dropna()
df1 = df.drop(columns = [ 'slope', 'thal', 'fbs', 'restecg','sex'])

df1.head()

correlat=df1.corr()
pd.DataFrame(correlat['target']).sort_values(by='target',ascending=False)
#df.head()

plt.figure(figsize=(8,5))
sns.heatmap(df1.corr(),linewidths=0.3, linecolor='black',annot_kws={"size": 10},annot=True,cmap="winter")
plt.show()
plt.savefig('correlationfigure')

from pandas.core.groupby.generic import DataFrameGroupBy
categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

categorical_val

df_dummy = pd.get_dummies(df, columns = categorical_val)
df_dummy.head()

df_dummy = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

#for continous value
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


s_sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df_dummy[col_to_scale] = s_sc.fit_transform(df_dummy[col_to_scale])

df_dummy.head()

train = df1.drop('target', axis = 1)
target = df1.target


X_train, X_test, y_train, y_test = train_test_split( train, target, test_size = 0.2, random_state = 45 )

print(X_test.head())

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model1 = LogisticRegression()
model1.fit(X_train,y_train)
Y_pred1 = model1.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix

cmdt= confusion_matrix(y_test,Y_pred1)
sns.heatmap(cmdt, annot=True,cmap='winter')
print(classification_report(y_test, Y_pred1))

TP=cmdt[0][0]
TN=cmdt[1][1]
FN=cmdt[1][0]
FP=cmdt[0][1]
#print(round(accuracy_score(Y_pred,y_test)*100,2))

print('Testing Accuracy for Logistic Regression:',(TP+TN)/(TP+TN+FN+FP))
print('Testing Sensitivity for Logistic Regression:',(TP/(TP+FN)))
print('Testing Specificity for Logistic Regression:',(TN/(TN+FP)))
print('Testing Precision for Logistic Regression:',(TP/(TP+FP)))

#Decision tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score


tree_model = DecisionTreeClassifier(max_depth=5,criterion='entropy')
cv_scores = cross_val_score(tree_model, train, target, cv=10, scoring='accuracy')
m=tree_model.fit(X_train,y_train)
prediction=m.predict(X_test)
cm= confusion_matrix(y_test,prediction)
sns.heatmap(cm, annot=True,cmap='winter',linewidths=0.3, linecolor='black',annot_kws={"size": 20})
print(classification_report(y_test, prediction))

TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print('Testing Accuracy for Decision Tree:',(TP+TN)/(TP+TN+FN+FP))
print('Testing Sensitivity for Decision Tree:',(TP/(TP+FN)))
print('Testing Specificity for Decision Tree:',(TN/(TN+FP)))
print('Testing Precision for Decision Tree:',(TP/(TP+FP)))

#RANDOM FORSET
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500,criterion='entropy',max_depth=8,min_samples_split=5)
model3 = rfc.fit(X_train, y_train)
prediction3 = model3.predict(X_test)
cm3=confusion_matrix(y_test, prediction3)
sns.heatmap(cm3, annot=True,cmap='winter',linewidths=0.3, linecolor='black',annot_kws={"size": 20})
TP=cm3[0][0]
TN=cm3[1][1]
FN=cm3[1][0]
FP=cm3[0][1]
print(round(accuracy_score(prediction3,y_test)*100,2))
print('Testing Accuracy for Random Forest:',(TP+TN)/(TP+TN+FN+FP))
print('Testing Sensitivity for Random Forest:',(TP/(TP+FN)))
print('Testing Specificity for Random Forest:',(TN/(TN+FP)))
print('Testing Precision for Random Forest:',(TP/(TP+FP)))

def model_use(report_data):
  s=" "


  as_numpy=np.array(report_data)
  reshaped=as_numpy.reshape(1,-1)
  pre1=tree_model.predict(reshaped)
  if(pre1==0):
    s+="The patient seems to be have heart disease:(......."+ "Please Contact your nearset Hospital for treatment..."
    print(s)
  # report.write(str1)
  # report.close()
  else:
    s+="The patient seems to be Normal:)"
    print(s)



#Chatbot

import string #library for string operations
import random  #to get response from the set of responses
import nltk
import numpy as np

overview_heart=open("what_heart.txt",'r',errors='ignore')
risk_heart=open("risk_heart.txt",'r',errors='ignore')
symptom_heart=open("symptom_heart.txt",'r',errors='ignore')
emergency_heart=open("whena-doctor_heart.txt",'r',errors='ignore')
#data=open("data_prediction.txt",'r',errors='ignore')
#report = open("report_file.txt",'r',errors='ignore')

corpus1=overview_heart.read()
c2=risk_heart.read()
c3=symptom_heart.read()
c4=emergency_heart.read()
#c5=data.read()
#c6=report.read()



#pre-processing text data

cf1=corpus1.lower()
cf2=c2.lower()
cf3=c3.lower()
cf4=c4.lower()
#cf5=c5.lower()
#cf6=c6.lower()



nltk.download('punkt')#using the punkit tokenizer
nltk.download('wordnet')#using the wordnet dicctinorary
nltk.download('omw-1.4')
# corpus=["cf1","cf2","cf3","cf4"]
# corpus

sentence_tokens=nltk.sent_tokenize(cf1)
word_tokens=nltk.word_tokenize(cf1)
sentence_tokens=nltk.sent_tokenize(cf2)
word_tokens=nltk.word_tokenize(cf2)
sentence_tokens=nltk.sent_tokenize(cf3)
word_tokens=nltk.word_tokenize(cf3)
sentence_tokens=nltk.sent_tokenize(cf4)
word_tokens=nltk.word_tokenize(cf4)
# sentence_tokens=nltk.sent_tokenize(cf5)
# word_tokens=nltk.word_tokenize(cf5)
# sentence_tokens=nltk.sent_tokenize(cf6)
# word_tokens=nltk.word_tokenize(cf6)

#1. removing punctuation
lemmer=nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punc_dict=dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

#greeting
greet_inputs=('helo',"hello","i want to check my report")
greet_response=("hi! Welcome to the Heart-Bot, your personal Heart disaese Predictor Bot------------")
def greet(sentence):
  for word in sentence.split():
    if word.lower() in greet_inputs:
      return random.choice(greet_response)

#intelligence or response of bot
  #1. feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer #using tfidf
from sklearn.metrics.pairwise import cosine_similarity #cosine similarity

#response
def response(user_response):
  bot_response=' '
  TfidfVec =TfidfVectorizer(tokenizer= LemNormalize, stop_words ='english')
  tfidf=TfidfVec.fit_transform(sentence_tokens)
  #cosine similarity
  vals=cosine_similarity(tfidf[-1], tfidf)
  idx=vals.argsort()[0][-2]
  flat=vals.flatten()
  flat.sort()
  req_tfidf=flat[-2]
  if(req_tfidf == 0):
    bot_response=bot_response+"I am sorry, unable to understand you!"
    return bot_response
  else:
      bot_response=bot_response+sentence_tokens[idx]
      return bot_response

flag = True
print( "Please! ask your doubt if any!!")
while(flag == True):
  user_response = input()
  user_response = user_response.lower()
  if(user_response != "bye"):
    if(user_response == 'thank you' or user_response == 'thanks'):
      flag= False
      print("Bot: You are Welcome..")
    else:
       if(greet(user_response) != None):
          print('Bot '+ greet(user_response))
       else:
         sentence_tokens.append(user_response)
         word_tokens = word_tokens + nltk.word_tokenize(user_response)
         final_words = list(set(word_tokens))
         print("Bot: ", end = "")
         print(response(user_response))
         sentence_tokens.remove(user_response)
  else:
       flag = False
       print('Bot: Goodbye!')

#Defining chatflow
print("Bot: Hello! I am Heart Disease Prediction Bot, Please Provide your name. :-")
flag = False
counter = 0
while not flag:
  userresponse = input()
  print("Bot: Please provide the following details from the Heart report:\n" + "Bot: 1. Age")
  age = int(input())
  print("Bot: 2. Chest pain value according to the following four categories:\n" + "  value 1 = typical angina\n  value 2 = atypical angina\n   value 3 = non — anginal pain\n   value 4 = asymptomatic" )
  cp = int(input())
  print("Bot: 3. trestbps (Resting Blood Pressure) in mmHg (unit)")
  tresbps = int(input())
  print("Bot: 4. Chol (Serum Cholesterol) in mg/dL (unit)")
  chol = int(input())
  print("Bot: 5. Thalach (Max Heart Rate Achieved)")
  thalach = int(input())
  print("Bot: 6. Exang: Exercise induced angina (1 = yes; 0 = no)")
  exang=int(input())
  print("Bot: 7. Oldpeak (ST depression induced by exercise relative to rest)")
  oldpeak = float(input())
  print("Bot: 8. ca (Number of major vessels (0–3) colored by fluoroscopy)")
  ca = int(input())
  report_data = (age, cp, tresbps, chol, thalach, exang, oldpeak, ca)
  #report_data= tuple(data)
  model_use(report_data)
  print("Thank you >>>>")
