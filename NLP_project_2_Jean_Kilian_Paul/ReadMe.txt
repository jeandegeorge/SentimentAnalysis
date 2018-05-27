Students:
Jean Degeorge - B00714740@essec.edu
Kilian Tep - B00720561@essec.edu
Paul Hayat - B00721702@essec.edu

With the file classifier_bow_full.py, we get an average prediction accuracy score of 79% on the dev set using a Logistic Regression model.

With the file classifier_bow.py, where all the unnecessary feature engineering was removed, we curiously get a slightly lower score. The computation time is shorter however.

To build our model, we cut each sentence into fragments using any word of similar type to 'but' as a cut-off. We then used a frequency BOW representation for words in the section of the cut-up sentence containing the target word. We finally used polarizer variables thank to the NLTK package, Vader, and by using a publicly available UCI lexicon (pos_words.xlsx and neg_words.xlsx).

The following methods were also used but were found not to be as predictive in our model:
- bi-grams and tri-grams
- POS tag vector representations
- keeping words in window of 5 (distance in terms of number of words)
- using SpaCy dependency parser to only keep words having a dependency relation to the target word (the Spacy parser was found to not be extremely efficient for unstructured sentences such as the ones in this data set)
- punctuation vector representations

Code for these methods can be found in the deliverable.

Following is the code to generate punctuation indicators and trigrams. We did not include it in the program so as not to slow it down since we did not keep theses variables in our model.

, "although", "however", "even though", "except", "though", "whereas", "even if", "nevertheless",
               "nonetheless", "yet", "on the other hand"

# add punctuation indicators

    for i, j in enumerate([',', ';', ':', "-"]):
        df['punct' + str(i)] = [df.loc[k, 'sentence'].count(str(j)) for k in range(np.shape(df)[0])]

# generate trigrams

    import string
    from nltk import ngrams


    punctless = ["".join((char for char in df["sentence"][i] if char not in string.punctuation)) for i in
                 range(df.shape[0])]

    trigrams = [ngrams(punctless[i].lower().split(), 3) for i in range(df.shape[0])]

    tri = []
    for i in range(df.shape[0]):
        tri.append([grams for grams in trigrams[i]])

    df['tri'] = tri

    # vectorize trigrams
    vocab_tri = list(set(sum(tri, [])))

    tricount = []
    count = -1
    for i in vocab_tri:
        tricount.append([])
        count += 1
        for j in range(df.shape[0]):
            tricount[count].append(tri[j].count(i))

    j=-1
    for i in vocab_tri:
        j+=1
        df['tri_' + str(j)] = tricount[j]

