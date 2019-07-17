import os, sys, pickle
import numpy as np
import pandas as pd

# UPDATE: This can run either locally or remotely
#
# Suggestions for development:
# Test multiple models, use k-fold cross validation, select best model
#
# CURRENT TEST
#
# Logistic Regression (or not)
# Multinomial Naive Bayes
# Complement Naive Bayes

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

def load_pickle(path_to_pickle):
    with open(path_to_pickle, 'rb') as p:
         return pickle.load(p)

# performance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                    average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                              average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

def main():
    '''Test & Evaluate multiple models to select best option

    Considerations:
    + Single-feature
    + Multiple features
    + Count vectorizer - BoW/Tfidf
    + k-fold x-validation
    + Choice of classifier (LR, RF, etc)

    Metrics:
    + Accuracy
    + Variance
    + F1
    + Confusion matrix
    + Create graphics
    + Feature importance
    '''

    path_to_module = '../tools/'
    sys.path.append(path_to_module)
    # load s3 read & write functions
    import bototools as bt
    import nlp_preprocessing as nlp

    print('classifying new jobs...\n')

    # if pulling from s3
    #path_to_data = '../.keys/'
    #file_name = 'csv_to_classify.json'
    #bucket, key = bt.load_s3_location(path_to_data, file_name)
    #df = bt.load_df_from_s3(bucket, key, compression='gzip')

    # local data
    path_to_data = '../data/multiclass/'
    data_file = 'labeled_large.csv'
    df = pd.read_csv(os.path.join(path_to_data, data_file))
    print('loaded CSV')

    # map roles to general labels
    key_path = '../.keys'
    role_mapper = load_pickle(os.path.join(key_path, 'roles.pickle'))

    role_names = [n for n in role_mapper]
    df = df[df['role'].isin(role_names)]

    df['label'] = df['role'].map(role_mapper)
    df = df.dropna()

    cols_to_train = ['title', 'label']
    df = df[cols_to_train]

    # standardize text format
    df = nlp.standardize_text(df, 'title')

    ## run a 10-fold stratified cross validation

    skf = StratifiedKFold(n_splits=10, random_state=0)
    
    df = df.reset_index(drop=True)
    X = df['title'].values.astype('U')
    y = df['label']

    current_split = 1
    acc_lr = []; prec_lr = []; rec_lr = []; f1_lr = []
    acc_mnb = []; prec_mnb = []; rec_mnb = []; f1_mnb = []
    acc_cnb = []; prec_cnb = []; rec_cnb = []; f1_cnb = []

    for train_index, test_index in skf.split(X, y):
        print('CURRENT SPLIT:', current_split)
    
        # get splits & assign data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print('successful split')

        # vectorize word counts
        X_train_counts, count_vectorizer = nlp.create_count_vectorizer(X_train)
        X_test_counts = count_vectorizer.transform(X_test)
        
        # Logistic regression
        lr = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
        lr.fit(X_train_counts, y_train)
        lr_predictions = lr.predict(X_test_counts)
        accuracy, precision, recall, f1 = get_metrics(y_test, lr_predictions)
        print("LR: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

        acc_lr.append(accuracy)
        prec_lr.append(precision)
        rec_lr.append(recall)
        f1_lr.append(f1)
    
        # multinomial NB
        mnb = MultinomialNB().fit(X_train_counts, y_train)
        mnb_predictions = mnb.predict(X_test_counts)
        accuracy, precision, recall, f1 = get_metrics(y_test, mnb_predictions)
        print("MNB: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
    
        acc_mnb.append(accuracy)
        prec_mnb.append(precision)
        rec_mnb.append(recall)
        f1_mnb.append(f1)
    
        # complement NB
        cnb = ComplementNB().fit(X_train_counts, y_train)
        cnb_predictions = cnb.predict(X_test_counts)
        accuracy, precision, recall, f1 = get_metrics(y_test, cnb_predictions)
        print("CNB: accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
    
        acc_cnb.append(accuracy)
        prec_cnb.append(precision)
        rec_cnb.append(recall)
        f1_cnb.append(f1)

        current_split += 1
    
    # Summarize
    print('\nFinal Perfomance')
    
    print('\nLogistic Regression Perfomance')
    print('Accuracy: mean %.3f, variance %.3f' % (np.mean(acc_lr), np.var(acc_lr)))
    print('Precision: mean %.3f, variance %.3f' % (np.mean(prec_lr), np.var(prec_lr)))
    print('Recall: mean %.3f, variance %.3f'% (np.mean(rec_lr), np.var(rec_lr)))
    print('F1 Score: mean %.3f, variance %.3f'% (np.mean(f1_lr), np.var(f1_lr)))

    print('\nMultinomial Naive Bayes Perfomance')
    print('Accuracy: mean %.3f, variance %.3f' % (np.mean(acc_mnb), np.var(acc_mnb)))
    print('Precision: mean %.3f, variance %.3f' % (np.mean(prec_mnb), np.var(prec_mnb)))
    print('Recall: mean %.3f, variance %.3f'% (np.mean(rec_mnb), np.var(rec_mnb)))
    print('F1 Score: mean %.3f, variance %.3f'% (np.mean(f1_mnb), np.var(f1_mnb)))
    
    print('\nComplement Naive Bayes Perfomance')
    print('Accuracy: mean %.3f, variance %.3f' % (np.mean(acc_cnb), np.var(acc_cnb)))
    print('Precision: mean %.3f, variance %.3f' % (np.mean(prec_cnb), np.var(prec_cnb)))
    print('Recall: mean %.3f, variance %.3f'% (np.mean(rec_cnb), np.var(rec_cnb)))
    print('F1 Score: mean %.3f, variance %.3f'% (np.mean(f1_cnb), np.var(f1_cnb)))


if __name__ == '__main__':
    main()
