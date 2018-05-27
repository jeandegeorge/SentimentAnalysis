import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_selection import SelectKBest, chi2
import random
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

random.seed(123)


class Classifier:
    """The Classifier"""

    def __init__(self):
        self.clf = None
        self.df = None
        self.kbest = None

    def feature_selection(self, dataset):
        df = dataset
        cols_w1 = [c for c in df.columns if c.lower()[:1] == '_']

        cols_w = cols_w1 + ['pol_num']
        cols_w = list(set(cols_w))
        df = df[cols_w]
        return df


    def load_and_clean(self,data_path):
        # read data set
        df = pd.read_csv(data_path, sep='\t', header=None)
        df.columns = ['polarity', 'aspect_cat', 'target', 'offsets', 'sentence']

        # create label dictionaries to create column with category number

        aspect_lab = {"AMBIENCE#GENERAL": 0, "DRINKS#PRICES": 1, "DRINKS#QUALITY": 2, "DRINKS#STYLE_OPTIONS": 3,
                      "FOOD#PRICES": 4, "FOOD#QUALITY": 5, "FOOD#STYLE_OPTIONS": 6, "LOCATION#GENERAL": 7,
                      "RESTAURANT#GENERAL": 8,
                      "RESTAURANT#MISCELLANEOUS": 9, "RESTAURANT#PRICES": 10, "SERVICE#GENERAL": 11}

        # do the same thing for polarity label

        pol_lab = {"positive": 0, "neutral": 1, "negative": 2}

        # create aspect category number column

        df.insert(loc=2, column='cat_num', value=df['aspect_cat'].map(aspect_lab))

        # create polarity category number column

        df.insert(loc=1, column='pol_num', value=df['polarity'].map(pol_lab))

        # create start and end column for indices of target words in sentence

        df.insert(loc=6, column='start_end', value=[df.loc[i, "offsets"].split(':') for i in range(np.shape(df)[0])])
        df["start"] = [int(df.loc[i, "start_end"][0]) for i in range(np.shape(df)[0])]
        df["end"] = [int(df.loc[i, "start_end"][1]) for i in range(np.shape(df)[0])]

        # split aspect categories and create column for each aspect category type

        df["aspect_cat"] = [df.loc[i, "aspect_cat"].split("#") for i in range(np.shape(df)[0])] # we split the 2 words.

        df["cat1"] = [df.loc[i, "aspect_cat"][0] for i in range(np.shape(df)[0])]
        aspect_lab1 = {"AMBIENCE": 0, "DRINKS": 1, "FOOD": 2, "LOCATION": 3, "RESTAURANT": 4, "SERVICE": 5}
        df['cat1'] = df['cat1'].map(aspect_lab1) # we fill 'cat2' with the numbers corresponding to each category.

        df["cat2"] = [df.loc[i, "aspect_cat"][1] for i in range(np.shape(df)[0])]
        aspect_lab2 = {"GENERAL": 0, "PRICES": 1, "QUALITY": 2, "STYLE_OPTIONS": 3, "MISCELLANEOUS": 4}
        df['cat2'] = df['cat2'].map(aspect_lab2) # we fill 'cat2' with the numbers corresponding to each category. 

        # chop up sentences

        cut = ["but"]

        df["sentence_cut"] = df["sentence"]

        for i in range(np.shape(df)[0]): # for every row of the dataset. 
            for c in cut:
                df.loc[i, "sentence_cut"] = df.loc[i, "sentence_cut"].replace(c, 'but') # replace all the cut words by 'but'.

        df["sentence_cut"] = [df.loc[i, "sentence_cut"].split("but") for i in range(np.shape(df)[0])] # split each sentence at the but.

        list_sent = [] # create a list of all sentences.
        for i in range(np.shape(df)[0]): # for every row of the dataset.
            list_sent.append([])
            for j in range(len(df.loc[i, "sentence_cut"])): # for each word of the sentence.
                if df.loc[i, "target"] in df.loc[i, "sentence_cut"][j]:
                    list_sent[i].append(df.loc[i, "sentence_cut"][j]) 

        for i in range(np.shape(df)[0]):
            list_sent[i] = " ".join(list_sent[i]) 

        # list_sent = sum(list_sent, [])

        df["list_sent"] = list_sent

        ### Tokenization: we will generate BOW and ngrams for the whole sentences, then for the dependencies,
        ### and finally, for the words in the window of size 5 (distance from the target).
        ### We will also assign sentiment scores to each of these representations of the text.
        ### We first delete stop words but will test a model keeping them.
        ### In addition, to word representations, we will generate POS variables

        # load SpaCy for English

        nlp = spacy.load('en_core_web_sm')

        # number of positive words among words which depend of target

        pos_words = pd.read_excel('/Users/philippehayat/Desktop/pos_words.xlsx')
        pos_words = list(pos_words.iloc[:, 0])

        # number of negative words among words which depend of target

        neg_words = pd.read_excel('/Users/philippehayat/Desktop/neg_words.xlsx')
        neg_words = list(neg_words.iloc[:, 0])

        # create lemmatized list of words from whole sentence as well as list of pos

        lemma_list = []
        for i in range(np.shape(df)[0]):
            lemma_list.append([])
            for token in nlp(df.loc[i, 'list_sent']):
                if (token.is_punct == False) & (token.is_alpha == True):
                    lemma_list[i].append(token.lemma_) # each words of every sentence of 'list_sent' is lemmatized. 

        # number of positive words among all words in sentence

        pos_score2 = []
        for i in range(np.shape(df)[0]):
            pos_score2.append([])
            for j in lemma_list[i]:
                if j in pos_words:
                    pos_score2[i].append(1)

        pos_score2 = [sum(pos_score2[i]) for i in range(len(pos_score2))] # we count the number of positive words in each sentence.

        # number of negative words among all words in sentence

        neg_score2 = []
        for i in range(np.shape(df)[0]):
            neg_score2.append([])
            for j in lemma_list[i]:
                if j in neg_words:
                    neg_score2[i].append(1)

        neg_score2 = [sum(neg_score2[i]) for i in range(len(neg_score2))] # we count the number of negative words in each sentence.

        # create vocab set and POS set for 3 methods of selecting words

        vocab_all = list(set(sum(lemma_list, [])))
        #pos_cat_all = list(set(sum(pos_list, [])))

        # assign to columns

        df['lemma_list'] = lemma_list
        #df['pos_list'] = pos_list

        df['sentence2'] = [" ".join(df['lemma_list'][i]) for i in range(np.shape(df)[0])]
        #df['pos_list2'] = [" ".join(df['pos_list'][i]) for i in range(np.shape(df)[0])]

        df['pos_score2'] = pos_score2

        df['neg_score2'] = neg_score2

        # BOW

        # BOW for all
        for i in vocab_all:
            df['_' + i] = df.sentence2.str.count(i)
        #for i in pos_cat_all:
            #df['_pos_' + i] = df.pos_list2.str.count(i)

        # Vader polarizer variable

        vader = SentimentIntensityAnalyzer() # vader gives a score between -1 and 1 to each sentence (that is why + 1).
        vader_all = [vader.polarity_scores(df['sentence2'][i])['compound'] + 1 for i in range(df.shape[0])]
        df['vader_all'] = vader_all

        return df

    def train(self, trainfile):
        """Trains the classifier model on the training set stored in file trainfile"""
        self.df = self.load_and_clean(trainfile)
        y = self.df['pol_num']
        y = y.astype(int)

        df_copy = self.df[:]
        self.df = self.feature_selection(self.df)

        X = self.df.drop(['pol_num'],axis=1)
        X = X.astype(float)

        self.kbest = SelectKBest(chi2, k=20).fit(X, y)
        X_new = self.kbest.transform(X)
        X_new = np.append(X_new, np.reshape(np.array(df_copy['vader_all']), (np.shape(X_new)[0], 1)), axis=1) # we add the column 'vader_all' to X_new.
        X_new = np.append(X_new, np.array(df_copy[['pos_score2', 'neg_score2']]), axis=1) # we add the columns 'pos_score2' and 'neg_score2' to X_new.

        param_grid = {'penalty':['l1', 'l2'],
        'C':[0.5, 1, 1.5, 1.7, 2, 2.4, 2.7],
        'fit_intercept':[False, True], 
        'class_weight':[None, 'balanced']}

        self.clf = LogisticRegression()
        self.grid = GridSearchCV(self.clf, param_grid, cv=2)
        self.grid.fit(X_new, y)
        self.params = self.grid.best_params_

        self.clf = LogisticRegression(**self.params)
        self.clf = self.clf.fit(X_new, y)
        

        #self.clf = RandomForestClassifier()
        #self.clf.fit(X_new, y) # train the model.

        print("accuracy score on train set", accuracy_score(y, self.clf.predict(X_new)))
        print("cross validation accuracy score on training set", np.mean(cross_val_score(self.clf, X_new, y, cv=5, scoring='accuracy')))

    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        df_test = self.load_and_clean(datafile)
        df_test_copy = df_test[:]
        df_test = self.feature_selection(df_test)

        # target
        y_test = df_test['pol_num']
        y_test = y_test.astype(int)

        # cols in train
        cols_w_dev = [c for c in self.df.columns if c in df_test.columns]
        cols_w_dev = list(set(cols_w_dev))
        df_test = df_test[cols_w_dev]

        cols_not = [c for c in self.df.columns if c not in df_test.columns]
        cols_not = list(set(cols_not))

        for c in cols_not:
            df_test[c] = 0

        X_test = df_test.drop(['pol_num'],axis=1)
        X_test = X_test.astype(float)

        X_test_new = self.kbest.transform(X_test)
        X_test_new = np.append(X_test_new, np.reshape(np.array(df_test_copy['vader_all']), (np.shape(X_test_new)[0], 1)), axis=1)
        X_test_new = np.append(X_test_new, np.array(df_test_copy[['pos_score2', 'neg_score2']]), axis=1)

        y_pred = self.clf.predict(X_test_new)
        print("Classifier acc:",accuracy_score(y_test, y_pred))

        y_pred_label = []

        for pred in y_pred:
            if pred==0:
                y_pred_label.append("positive")
            elif pred==1:
                y_pred_label.append("neutral")
            elif pred==2:
                y_pred_label.append("negative")

        return y_pred_label


