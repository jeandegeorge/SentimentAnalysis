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
        df=dataset
        cols_w1 = [c for c in df.columns if c.lower()[:1] == '_']
        cols_w2 = [c for c in df.columns if c.lower()[:5] == '_pos_']
        cols_w3 = [c for c in df.columns if c.lower()[3:9] == '_score']  # positivity and negativity scores
        cols_w4 = [c for c in df.columns if c.lower()[:2] == 'w_']
        cols_w5 = [c for c in df.columns if c.lower()[:6] == 'pos_w_']
        cols_w6 = [c for c in df.columns if c.lower()[:2] == '__']
        cols_w7 = [c for c in df.columns if c.lower()[:5] == 'pos__']

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

        df["aspect_cat"] = [df.loc[i, "aspect_cat"].split("#") for i in range(np.shape(df)[0])]

        df["cat1"] = [df.loc[i, "aspect_cat"][0] for i in range(np.shape(df)[0])]
        aspect_lab1 = {"AMBIENCE": 0, "DRINKS": 1,
                       "FOOD": 2, "LOCATION": 3,
                       "RESTAURANT": 4, "SERVICE": 5}
        df['cat1'] = df['cat1'].map(aspect_lab1)

        df["cat2"] = [df.loc[i, "aspect_cat"][1] for i in range(np.shape(df)[0])]
        aspect_lab2 = {"GENERAL": 0, "PRICES": 1, "QUALITY": 2, "STYLE_OPTIONS": 3, "MISCELLANEOUS": 4}
        df['cat2'] = df['cat2'].map(aspect_lab2)

        # chop up sentences

        cut = ["but", "although", "however", "even though", "except", "though", "whereas", "even if", "nevertheless",
               "nonetheless", "yet", "on the other hand"]

        df["sentence_cut"] = df["sentence"]

        for i in range(np.shape(df)[0]):
            for c in cut:
                df.loc[i, "sentence_cut"] = df.loc[i, "sentence_cut"].replace(c, 'but')

        df["sentence_cut"] = [df.loc[i, "sentence_cut"].split("but") for i in range(np.shape(df)[0])]

        list_sent = []
        for i in range(np.shape(df)[0]):
            list_sent.append([])
            for j in range(len(df.loc[i, "sentence_cut"])):
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

        # create set of stop words

        stop = set(stopwords.words('english'))

        # tokenize target

        target_tokens = []
        for i in range(np.shape(df)[0]):
            target_tokens.append([])
            for token in nlp(df.loc[i, 'target']):
                target_tokens[i].append(str(token))

        # make list of words that have relation of dependency to target and make a list of POS of target

        word_list = []
        target_pos = []  # list of target + dependent pos
        for i in range(np.shape(df)[0]):
            word_list.append([])
            target_pos.append([])
            for token in nlp(df.loc[i, 'list_sent']):
                if (str(token.text) in target_tokens[i]):
                    for sub in token.subtree:
                        if (sub.is_alpha == True) & (sub.is_punct == False) & (str(sub.lemma_) not in stop):
                            word_list[i].append(str(sub.lemma_))
                        target_pos[i].append(sub.pos_)
                    for ancestor in token.ancestors:
                        if (ancestor.is_alpha == True) & (ancestor.is_punct == False) & (str(ancestor.lemma_) not in stop):
                            word_list[i].append(str(ancestor.lemma_))
                        target_pos[i].append(ancestor.pos_)
                    target_pos[i].append(token.pos_)
            word_list[i] = list(set(word_list[i]))

        # number of positive words among words which depend of target

        pos_words = pd.read_excel('pos_words.xlsx')
        pos_words = list(pos_words.iloc[:, 0])

        pos_score = []
        for i in range(np.shape(df)[0]):
            pos_score.append([])
            for j in word_list[i]:
                if j in pos_words:
                    pos_score[i].append(1)

        pos_score = [sum(pos_score[i]) for i in range(len(pos_score))]

        # number of negative words among words which depend of target

        neg_words = pd.read_excel('neg_words.xlsx')
        neg_words = list(neg_words.iloc[:, 0])

        neg_score = []
        for i in range(np.shape(df)[0]):
            neg_score.append([])
            for j in word_list[i]:
                if j in neg_words:
                    neg_score[i].append(1)

        neg_score = [sum(neg_score[i]) for i in range(len(neg_score))]

        # create lemmatized list words from whole sentence as well as list of pos

        lemma_list = []
        pos_list = []
        for i in range(np.shape(df)[0]):
            lemma_list.append([])
            pos_list.append([])
            for token in nlp(df.loc[i, 'list_sent']):
                if (token.is_punct == False) & (token.is_alpha == True):
                    lemma_list[i].append(token.lemma_)
                pos_list[i].append(token.pos_)

        # number of positive words among all words in sentence

        pos_score2 = []
        for i in range(np.shape(df)[0]):
            pos_score2.append([])
            for j in lemma_list[i]:
                if j in pos_words:
                    pos_score2[i].append(1)

        pos_score2 = [sum(pos_score2[i]) for i in range(len(pos_score2))]

        # number of negative words among all words in sentence

        neg_score2 = []
        for i in range(np.shape(df)[0]):
            neg_score2.append([])
            for j in lemma_list[i]:
                if j in neg_words:
                    neg_score2[i].append(1)

        neg_score2 = [sum(neg_score2[i]) for i in range(len(neg_score2))]

        # Create list of token indices within the window to create window instead of using dependency

        token_i = []
        for i in range(np.shape(df)[0]):
            token_i.append([])
            for token in nlp(df.loc[i, 'list_sent']):
                if str(token) in target_tokens[i]:
                    token_i[i].append(token.i)

        context_indices = []
        for i in range(np.shape(df)[0]):
            context_indices.append([])
            for token in nlp(df.loc[i, 'list_sent']):
                for j in token_i[i]:
                    if token.i in range(j - 7, j + 7):
                        context_indices[i].append(token.i)
            context_indices[i] = list(set(context_indices[i]))

        # create list of words and pos in window

        lemma_list_w = []
        pos_list_w = []
        for i in range(np.shape(df)[0]):
            lemma_list_w.append([])
            pos_list_w.append([])
            for token in nlp(df.loc[i, 'list_sent']):
                if (token.is_punct == False) & (token.is_alpha == True) & (token.i in context_indices[i]):
                    lemma_list_w[i].append(token.lemma_)
                pos_list_w[i].append(token.pos_)

        # (str(token.lemma_) not in stop)

        # number of positive words among all words in sentence

        pos_score3 = []
        for i in range(np.shape(df)[0]):
            pos_score3.append([])
            for j in lemma_list_w[i]:
                if j in pos_words:
                    pos_score3[i].append(1)

        pos_score3 = [sum(pos_score3[i]) for i in range(len(pos_score3))]

        # number of negative words among all words in sentence

        neg_score3 = []
        for i in range(np.shape(df)[0]):
            neg_score3.append([])
            for j in lemma_list_w[i]:
                if j in neg_words:
                    neg_score3[i].append(1)

        neg_score3 = [sum(neg_score3[i]) for i in range(len(neg_score3))]

        # create vocab set and POS set for 3 methods of selecting words

        vocab_all = list(set(sum(lemma_list, [])))
        pos_cat_all = list(set(sum(pos_list, [])))

        vocab_w = list(set(sum(lemma_list_w, [])))
        pos_cat_w = list(set(sum(pos_list_w, [])))

        vocab_dep = list(set(sum(word_list, [])))
        pos_cat_dep = list(set(sum(target_pos, [])))

        # assign to columns

        df['lemma_list'] = lemma_list
        df['pos_list'] = pos_list

        df['word_list'] = word_list
        df['target_pos'] = target_pos

        df['lemma_list_w'] = lemma_list_w
        df['pos_list_w'] = pos_list_w

        df['sentence2'] = [" ".join(df['lemma_list'][i]) for i in range(np.shape(df)[0])]
        df['pos_list2'] = [" ".join(df['pos_list'][i]) for i in range(np.shape(df)[0])]

        df['sentence_dep'] = [" ".join(df['word_list'][i]) for i in range(np.shape(df)[0])]
        df['pos_list2_dep'] = [" ".join(df['target_pos'][i]) for i in range(np.shape(df)[0])]

        df['sentence3'] = [" ".join(df['lemma_list_w'][i]) for i in range(np.shape(df)[0])]
        df['pos_list2_w'] = [" ".join(df['pos_list_w'][i]) for i in range(np.shape(df)[0])]

        df['pos_score'] = pos_score
        df['pos_score2'] = pos_score2
        df['pos_score3'] = pos_score3

        df['neg_score'] = neg_score
        df['neg_score2'] = neg_score2
        df['neg_score3'] = neg_score3

        # BOW

        # BOW for all
        for i in vocab_all:
            df['_' + i] = df.sentence2.str.count(i)
        for i in pos_cat_all:
            df['_pos_' + i] = df.pos_list2.str.count(i)

        # BOW for window
        for i in vocab_w:
            df['w_' + i] = df.sentence3.str.count(i)
        for i in pos_cat_w:
            df['pos_w_' + i] = df.pos_list2_w.str.count(i)

        # sub BOW for dependency list
        for i in vocab_dep:
            df['__' + i] = df.sentence_dep.str.count(i)
        for i in pos_cat_dep:
            df['pos__' + i] = df.pos_list2_dep.str.count(i)
            # df['__' + i] = [1 if df['__' + i][j] > 0 else 0 for j in range(df.shape[0])]

        # Vader polarizer variable

        vader = SentimentIntensityAnalyzer()
        vader_dep = [vader.polarity_scores(df['sentence_dep'][i])['compound'] + 1 for i in range(df.shape[0])]
        vader_all = [vader.polarity_scores(df['sentence2'][i])['compound'] + 1 for i in range(df.shape[0])]
        vader_w = [vader.polarity_scores(df['sentence3'][i])['compound'] + 1 for i in range(df.shape[0])]

        df['vader_dep'] = vader_dep
        df['vader_all'] = vader_all
        df['vader_w'] = vader_w

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
        X_new = np.append(X_new, np.reshape(np.array(df_copy['vader_all']), (np.shape(X_new)[0], 1)), axis=1)
        X_new = np.append(X_new, np.array(df_copy[['pos_score2', 'neg_score2']]), axis=1)

        #self.clf = RandomForestClassifier()
        #self.clf.fit(X_new, y)

        param_grid = {'penalty':['l1', 'l2'],
        'C':[0.5, 1, 1.5, 1.7, 2, 2.4, 2.7],
        'fit_intercept':[False, True], 
        'class_weight':[None, 'balanced']}

        # Logistic Regression:
        self.clf = LogisticRegression()
        self.grid = GridSearchCV(self.clf, param_grid, cv=5)
        self.grid.fit(X_new, y)
        self.params = self.grid.best_params_

        self.clf = LogisticRegression(**self.params)
        self.clf = self.clf.fit(X_new, y)

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


