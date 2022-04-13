Vinícius Trainotti

# Exercício - sklearn-assignment

Analisando o que foi passado em aula com as bibliotecas que já estavam setadas no arquivo do exercício, verificamos que o intuito era utilizar essas bibliotecas: TfidfVectorizer, LinearSVC; pesquisando sobre elas, foi possível entender e comparar com o processo que fizemos em sala de aula e adaptar conforme a necessidade, porém não apresentava valores coerentes. Como foi deixado o autor do arquivo original, foi possível encontrar o github com o repositório original, assim foi possível executar e apresentar o primeiro valor coerente e usado como base/parâmetro de execução:

Saída LinearSVC com valor train_test com test_size 0.25

```sh
n_samples: 2000
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
