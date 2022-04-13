Vinícius Trainotti

# Exercício - sklearn-assignment

Analisando o que foi passado em aula com as bibliotecas que já estavam setadas no arquivos do exercício, verificamos que o intuito era utilizar essas bibliotecas: TfidfVectorizer, LinearSVC; pesquisando sobre elas, foi possível entender e comparar com o processo que fizemos em sala de aula e adaptar conforme a necessidade, porém não apresentava valores coerentes. Como foi deixado o autor do arquivo original, foi possível achar o github com o repositório do exercício, assim conseguindo executar e ter o primeiro valor de parâmetro de execução:

Saída Padrão com valor train_test com test_size 0.25

```sh
{'mean_fit_time': array([0.86058807, 3.07510657]), 'std_fit_time': array([0.12694251, 0.6299907 ]), 'mean_score_time': array([0.16521358, 0.2905344 ]), 'std_score_time': array([0.01421598, 0.06387942]), 'param_vect__ngram_range': masked_array(data=[(1, 1), (1, 2)],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'params': [{'vect__ngram_range': (1, 1)}, {'vect__ngram_range': (1, 2)}], 'split0_test_score': array([0.85666667, 0.87      ]), 'split1_test_score': array([0.82666667, 0.83666667]), 'split2_test_score': array([0.86      , 0.87666667]), 'split3_test_score': array([0.84333333, 0.85333333]), 'split4_test_score': array([0.83333333, 0.84333333]), 'mean_test_score': array([0.844, 0.856]), 'std_test_score': array([0.01289272, 0.0152607 ]), 'rank_test_score': array([2, 1], dtype=int32)}
0 params - {'vect__ngram_range': (1, 1)}; mean - 0.84; std - 0.01
1 params - {'vect__ngram_range': (1, 2)}; mean - 0.86; std - 0.02
              precision    recall  f1-score   support

         neg       0.87      0.84      0.85       254
         pos       0.84      0.87      0.85       246

    accuracy                           0.85       500
   macro avg       0.85      0.85      0.85       500
weighted avg       0.85      0.85      0.85       500

[[214  40]
 [ 33 213]]
```

Alterando para o Algoritmo para KNeighborsClassifier usando todos os valores default 
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

```sh
n_samples: 2000
{'mean_fit_time': array([0.60445409, 1.67843037]), 'std_fit_time': array([0.02336203, 0.07190287]), 'mean_score_time': array([0.24879999, 0.37834682]), 'std_score_time': array([0.00439196, 0.06314027]), 'param_vect__ngram_range': masked_array(data=[(1, 1), (1, 2)],
             mask=[False, False],
       fill_value='?',
            dtype=object), 'params': [{'vect__ngram_range': (1, 1)}, {'vect__ngram_range': (1, 2)}], 'split0_test_score': array([0.63      , 0.63666667]), 'split1_test_score': array([0.63, 0.65]), 'split2_test_score': array([0.65333333, 0.67      ]), 'split3_test_score': array([0.65333333, 0.67      ]), 'split4_test_score': array([0.63      , 0.64666667]), 'mean_test_score': array([0.63933333, 0.65466667]), 'std_test_score': array([0.01143095, 0.0132665 ]), 'rank_test_score': array([2, 1], dtype=int32)}
0 params - {'vect__ngram_range': (1, 1)}; mean - 0.64; std - 0.01
1 params - {'vect__ngram_range': (1, 2)}; mean - 0.65; std - 0.01
              precision    recall  f1-score   support

         neg       0.76      0.56      0.64       256
         pos       0.64      0.82      0.72       244

    accuracy                           0.68       500
   macro avg       0.70      0.69      0.68       500
weighted avg       0.70      0.68      0.68       500

[[143 113]
 [ 45 199]]
```

Considerando que os ajustes utilizados no LinearSVC e os valores default do KNN, é possível verificar que tem diferença pelo resultado do crossvalidation (matrix de confusão), e claro pela acurácia apresentada. Bem, com o teste de um classificador diferente pra ter uma base, como o nosso parametro era pelo LinearSVC, o resultado final tem grande diferença.