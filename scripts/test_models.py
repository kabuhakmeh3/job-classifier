import os, sys
import pandas as pd

# this was done locally in a jupyter notebook
# update this for re-evaluating models with larger datasets on aws
# Test multiple models, use k-fold cross validation, select best model
#
# CURRENT TEST
#
# Logistic Regression (or not)
# Multinomial Naive Bayes
# Complement Naive Bayes
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#from sklearn.naive_bayes import ComplementNB

#def evaluate_model(X, y, vectorizer, model_name, random_state=0, confusion=True):
#    '''Select X features when calling the function

#    Add k-fold, report variance and mean?

#    Add confustion Matrix

#    create pngs with figures?
#    '''

    # create train/test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    # vectorize X

    # get model
    #model = model_name
    #model.fit(X_train, y_train)
    #y_predicted = model.predict(X_test)

    # print metrics -- THESE NEED TO BE UPDATED
    #print('--- /// BASE MODEL /// ---')
    #print('--- Classification Report ---')
    #print(classification_report(y_test, y_predicted, labels=[0,1], target_names=['Job', 'Gig']))
    #print('--- Confusion Matrix ---')
    #print(confusion_matrix(y_test, y_predicted))

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
    # remove samples not of interestc
    exclude = ['crna', 'pharmacist', 'cna']
    df = df[~(df.isin(exclude))]

    # map roles to general labels
    role_mapper = {'delivery':'driver',
                   'server':'service',
                   'product manager':'ignore',
                   'software engineer':'tech',
                   'electrical engineer':'ignore',
                   'systems engineer':'ignore',
                   'mechanical engineer':'ignore',
                   'nurse':'nurse',
                   'business analyst':'ignore',
                   'cook':'service',
                   'chef':'service',
                   'devops engineer':'tech',
                   'machine learning engineer':'tech',
                   'registered nurse':'nurse',
                   'cdl':'ignore'}

    df['label'] = df['role'].map(role_mapper)
    df = df.dropna()

    cols_to_train = ['title', 'label']
    df = df[cols_to_train]

    # standardize text format
    df = nlp.standardize_text(df, 'title')

    # select data to predict from
    X = df['title'].tolist()
    y = df['label'].tolist()
    #y = df['gig'].tolist()

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    print('vectorizing bag of words model')
    X_train_counts, count_vectorizer = nlp.create_count_vectorizer(X_train) # count vec
    X_test_counts = count_vectorizer.transform(X_test)

    # training a Naive Bayes classifier
    print('testing model')
    mnb = MultinomialNB().fit(X_train_counts, y_train)
    mnb_predictions = mnb.predict(X_test_counts)

    # evaluate performance
    from sklearn.metrics import confusion_matrix
    roles = ['tech','nurse','service','driver','ignore']
    cm = confusion_matrix(y_test, mnb_predictions, labels=roles)
    print('--- Confusion Matrix ---')
    print(roles)
    print(cm)

    # save model for later use (locally & on s3)
    #file_to_write = 'full_model.pckl'
    #bt.write_df_to_s3(df_sample, bucket, file_to_write)

if __name__ == '__main__':
    main()
