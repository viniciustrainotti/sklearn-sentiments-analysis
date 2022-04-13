"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD
# https://github.com/scikit-learn/scikit-learn/blob/main/doc/tutorial/text_analytics/solutions/exercise_02_sentiment.py

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics


if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = r"./data"
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    text_clf = Pipeline([('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
                        ('clf', LinearSVC(C=1000)),
    ])

    text_clf.fit(docs_train, y_train)

    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  #'tfidf__use_idf': (True, False),
                  #'clf__alpha': (1e-2, 1e-3),
    }
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(docs_train, y_train)

    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    print(gs_clf.cv_results_)
    n_scores = len(gs_clf.cv_results_['params'])
    for i in range(n_scores):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
                 % (gs_clf.cv_results_['params'][i],
                    gs_clf.cv_results_['mean_test_score'][i],
                    gs_clf.cv_results_['std_test_score'][i]))

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = gs_clf.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.show()

# Analisando o que foi passado na aula com as bibliotecas que já estavam setadas no arquivo, foi somente estuda-las e entender o processo pra adequar ao que foi passado.
# Além de que o arquivo tinha o autor original, foi possível achar no github pra ter uma base do que era esperado de resultado, então somente modifiquei com os valores necessários e feito alguns testes.

# Sobre as funções a TfidfVectorizer é parecida com o CountVectorizer, com os parametros de ignorar por quantidade minima e maxima do intervalo ou contagem que voce definir

# LinearSVC parecido com o SVC que vimos em aula, porém ele por default vem o kernel com linear, da pra ver bem como usar pela documentação do sklearn
# https://scikit-learn.org/stable/modules/svm.html#svm-classification

# Resultados no arquivo

# Saída Padrão com valor train_test com test_size 0.25
# {'mean_fit_time': array([0.86058807, 3.07510657]), 'std_fit_time': array([0.12694251, 0.6299907 ]), 'mean_score_time': array([0.16521358, 0.2905344 ]), 'std_score_time': array([0.01421598, 0.06387942]), 'param_vect__ngram_range': masked_array(data=[(1, 1), (1, 2)],
#              mask=[False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'vect__ngram_range': (1, 1)}, {'vect__ngram_range': (1, 2)}], 'split0_test_score': array([0.85666667, 0.87      ]), 'split1_test_score': array([0.82666667, 0.83666667]), 'split2_test_score': array([0.86      , 0.87666667]), 'split3_test_score': array([0.84333333, 0.85333333]), 'split4_test_score': array([0.83333333, 0.84333333]), 'mean_test_score': array([0.844, 0.856]), 'std_test_score': array([0.01289272, 0.0152607 ]), 'rank_test_score': array([2, 1], dtype=int32)}
# 0 params - {'vect__ngram_range': (1, 1)}; mean - 0.84; std - 0.01
# 1 params - {'vect__ngram_range': (1, 2)}; mean - 0.86; std - 0.02
#               precision    recall  f1-score   support

#          neg       0.87      0.84      0.85       254
#          pos       0.84      0.87      0.85       246

#     accuracy                           0.85       500
#    macro avg       0.85      0.85      0.85       500
# weighted avg       0.85      0.85      0.85       500

# [[214  40]
#  [ 33 213]]

# Aumentando para 50% da amostra pra treinamento
# n_samples: 2000
# {'mean_fit_time': array([0.46051302, 1.25552239]), 'std_fit_time': array([0.01009971, 0.13023678]), 'mean_score_time': array([0.10508938, 0.17792854]), 'std_score_time': array([0.00227407, 0.03215268]), 'param_vect__ngram_range': masked_array(data=[(1, 1), (1, 2)],
#              mask=[False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'vect__ngram_range': (1, 1)}, {'vect__ngram_range': (1, 2)}], 'split0_test_score': array([0.8 , 0.86]), 'split1_test_score': array([0.835, 0.855]), 'split2_test_score': array([0.815, 0.835]), 'split3_test_score': array([0.77 , 0.815]), 'split4_test_score': array([0.785, 0.82 ]), 'mean_test_score': array([0.801, 0.837]), 'std_test_score': array([0.02267157, 0.01805547]), 'rank_test_score': array([2, 1], dtype=int32)}
# 0 params - {'vect__ngram_range': (1, 1)}; mean - 0.80; std - 0.02
# 1 params - {'vect__ngram_range': (1, 2)}; mean - 0.84; std - 0.02
#               precision    recall  f1-score   support

#          neg       0.87      0.85      0.86       509
#          pos       0.85      0.87      0.86       491

#     accuracy                           0.86      1000
#    macro avg       0.86      0.86      0.86      1000
# weighted avg       0.86      0.86      0.86      1000

# [[434  75]
#  [ 63 428]]

