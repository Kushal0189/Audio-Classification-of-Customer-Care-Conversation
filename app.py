#!/usr/bin/env python
# coding: utf-8

# <h1>Data preparation phase</h1>
# <h1>Training Data</h1>

# In[1]:


import os
import soundfile as sf
import speech_recognition as sr
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# In[2]:


train = pd.read_csv("data.csv")
train.head()


# In[3]:


#special character cleaning

# \r and \n
train['Content_Parsed_1'] = train['Content'].str.replace("\r", " ")
train['Content_Parsed_1'] = train['Content_Parsed_1'].str.replace("\n", " ")
train['Content_Parsed_1'] = train['Content_Parsed_1'].str.replace("    ", " ")


# In[4]:


train['Content_Parsed_1'] = train['Content_Parsed_1'].str.replace('"', '')


# In[5]:


# Lowercasing the text
train['Content_Parsed_2'] = train['Content_Parsed_1'].str.lower()


# In[6]:


punctuation_signs = list("?:!.,;")
train['Content_Parsed_3'] = train['Content_Parsed_2']

for punct_sign in punctuation_signs:
    train['Content_Parsed_3'] = train['Content_Parsed_3'].str.replace(punct_sign, '')


# In[7]:


train['Content_Parsed_4'] = train['Content_Parsed_3'].str.replace("'s", "")


# In[9]:


# Saving the lemmatizer into an object
wordnet_lemmatizer = WordNetLemmatizer()

nrows = len(train)
lemmatized_text_list = []

for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = train.loc[row]['Content_Parsed_4']
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)


# In[10]:


train['Content_Parsed_5'] = lemmatized_text_list


# In[11]:


stop_words = list(stopwords.words('english'))


# In[12]:


train['Content_Parsed_6'] = train['Content_Parsed_5']

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"
    train['Content_Parsed_6'] = train['Content_Parsed_6'].str.replace(regex_stopword, '')


# In[13]:


list_columns = [ "Category", "Content", "Content_Parsed_6"]
train = train[list_columns]
train = train.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
train.head()


# <h1>Testing data</h1>
# <h4>Converting audio to text</h4>

# In[14]:


r = sr.Recognizer()
files = ['New Vehicle Purchase','Test Drive','Feedback','Vehicle Quality','Break Down']

for file in files:
    save = sr.AudioFile(file+'.wav')

    f = sf.SoundFile(file+'.wav')
    f1 = open(file+".txt","w+")
    sec = int(len(f) / f.samplerate)
    for i in range(0,sec,5):
        with save as source: 
            a = r.adjust_for_ambient_noise(source,duration=i)
            a = r.record(source)
            print(r.recognize_google(a,language='en-IN', show_all = True), file = f1)
    f1.close()


# In[109]:


import pandas as pd
test = pd.DataFrame(columns=['Content','Category'])
files = ['New Vehicle Purchase','Test Drive','Feedback','Vehicle Quality','Break Down']

Content = list()
Category = list()
for file in files:
    filename = file+'.txt'
    f = open(filename, 'rt')
    text = f.read()
    Content.append(text)
    Category.append(file)
    f.close()
    
list_of_tuples = list(zip(Content, Category))
test = pd.DataFrame(list_of_tuples, columns = ['Content', 'Category'])
test.to_csv('test.csv')
print(test)


# In[110]:


#special character cleaning

# \r and \n
test['Content_Parsed_1'] = test['Content'].str.replace("\r", " ")
test['Content_Parsed_1'] = test['Content_Parsed_1'].str.replace("\n", " ")
test['Content_Parsed_1'] = test['Content_Parsed_1'].str.replace("    ", " ")


# In[111]:


test['Content_Parsed_1'] = test['Content_Parsed_1'].str.replace('"', '')


# In[112]:


# Lowercasing the text
test['Content_Parsed_2'] = test['Content_Parsed_1'].str.lower()


# In[113]:


punctuation_signs = list("?:!.,;''[]{}")
remove_word = ['TRUE','alternative','transcript','confidence','final']
test['Content_Parsed_3'] = test['Content_Parsed_2']

for punct_sign in punctuation_signs:
    test['Content_Parsed_3'] = test['Content_Parsed_3'].str.replace(punct_sign, '')
for r_w in remove_word:
    test['Content_Parsed_3'] = test['Content_Parsed_3'].str.replace(r_w, '')


# In[114]:


test['Content_Parsed_4'] = test['Content_Parsed_3'].str.replace("'s", "")


# In[115]:


# Saving the lemmatizer into an object
wordnet_lemmatizer = WordNetLemmatizer()

nrows = len(test)
lemmatized_text_list = []

for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = test.loc[row]['Content_Parsed_4']
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)


# In[116]:


test['Content_Parsed_5'] = lemmatized_text_list


# In[117]:


stop_words = list(stopwords.words('english'))


# In[118]:


test['Content_Parsed_6'] = test['Content_Parsed_5']

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"
    test['Content_Parsed_6'] = test['Content_Parsed_6'].str.replace(regex_stopword, '')


# In[119]:


list_columns = [ "Category", "Content", "Content_Parsed_6"]
test = test[list_columns]
test = test.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
test.head()


# <h1>Label encoding</h1>

# In[120]:


category_codes = {
    'New Vehicle Purchase': 0,
    'Break Down': 1,
    'Feedback': 2,
    'Test Drive': 3,
    'Vehicle Quality': 4
}


# In[121]:


# train Category mapping
train['Category_Code'] = train['Category']
train = train.replace({'Category_Code':category_codes})


# In[122]:


# test Category mapping
test['Category_Code'] = test['Category']
test = test.replace({'Category_Code':category_codes})


# In[123]:


test.head()


# <h1>Traing and Testing</h1>

# In[30]:


X_train = train['Content']
y_train = train['Category_Code']
X_test = test['Content']
y_test = test['Category_Code']


# <h1>Text Representation</h1>

# In[31]:


tfidf = TfidfVectorizer()


# In[32]:


features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)


# In[33]:


from sklearn.feature_selection import chi2
import numpy as np

for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-10:])))
    print()


# <h1>Modeling</h1>
# <h2>Support Vector Machine</h2>

# In[34]:


from sklearn import svm
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
import pandas as pd


# In[35]:


svc_0 =svm.SVC(random_state=8)

print('Parameters currently in use:\n')
pprint(svc_0.get_params())


# In[36]:


# C
C = [.0001, .001, .01]

# gamma
gamma = [.0001, .001, .01, .1, 1, 10, 100]

# degree
degree = [1, 2, 3, 4, 5]

# kernel
kernel = ['linear', 'rbf', 'poly']

# probability
probability = [True]

# Create the random grid
random_grid = {'C': C,
              'kernel': kernel,
              'gamma': gamma,
              'degree': degree,
              'probability': probability
             }

pprint(random_grid)


# In[37]:


# First create the base model to tune
svc = svm.SVC(random_state=8)

# Definition of the random search
random_search = RandomizedSearchCV(estimator=svc,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   #cv=6, 
                                   verbose=1, 
                                   random_state=8)
labels_train = labels_train.astype('int')
# Fit the random search model
random_search.fit(features_train, labels_train)


# In[38]:


print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)


# In[39]:


# Create the parameter grid based on the results of random search 
C = [.0001, .001, .01, .1]
degree = [3, 4, 5]
gamma = [1, 10, 100]
probability = [True]

param_grid = [
  {'C': C, 'kernel':['linear'], 'probability':probability},
  {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability},
  {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
]

# Create a base model
svc = svm.SVC(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)


# In[40]:


print("The best hyperparameters from Grid Search are:")
print(grid_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)


# In[41]:


best_svc = random_search.best_estimator_


# In[42]:


best_svc


# In[43]:


best_svc.fit(features_train, labels_train)


# In[44]:


svc_pred = best_svc.predict(features_test)


# In[45]:


# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(labels_train, best_svc.predict(features_train)))


# In[46]:


# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(labels_test, svc_pred))


# In[47]:


# Classification report
print("Classification report")
print(classification_report(labels_test,svc_pred))


# In[48]:


base_model = svm.SVC(random_state = 8)
base_model.fit(features_train, labels_train)
accuracy_score(labels_test, base_model.predict(features_test))


# In[49]:


best_svc.fit(features_train, labels_train)
accuracy_score(labels_test, best_svc.predict(features_test))


# In[125]:


output = test.copy()
output = output.drop(['Content'], axis=1)
output['Predicted Values'] = svc_pred
output.rename(columns={'Category_Code':'Actual Values'}, inplace=True)
output.to_csv('Code_Scratchers_I-656A1.csv')



