import sys
import pandas as pd

# Test multiple models, use k-fold cross validation, select best model

def evaluate_model(X, y, vectorizer, model_name, random_state=0, confusion=True):
    '''Select X features when calling the function

    Add k-fold, report variance and mean?

    Add confustion Matrix

    create pngs with figures?
    '''

    # create train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    # vectorize X

    # get model
    model = model_name
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    # print metrics
    #print('--- /// BASE MODEL /// ---')
    print('--- Classification Report ---')
    print(classification_report(y_test, y_predicted, labels=[0,1], target_names=['Job', 'Gig']))
    print('--- Confusion Matrix ---')
    print(confusion_matrix(y_test, y_predicted))

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
    path_to_data = '../data/'
    data_file = 'labeled_eda_sample_data_file.csv'
    df = pd.read_csv(os.join.path(path_to_data, data_file))

    # standardize text format
    cols_to_model = ['title']
    for col in cols_to_model:
        df = nlp.standardize_text(df, col)

    # select data to predict from
    X = df['title'].tolist()
    y = df['gig'].tolist()

    # models to test
    # LogisticRegression, RandomForestClassifier, KNN

    # fit/transform vectorizer

    # fit/transform classifier

    # save model for later use (locally & on s3)
    #file_to_write = 'full_model.pckl'
    #bt.write_df_to_s3(df_sample, bucket, file_to_write)

if __name__ == '__main__':
    main()
