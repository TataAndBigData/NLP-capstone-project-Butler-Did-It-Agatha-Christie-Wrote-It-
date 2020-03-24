# NLP-capstone-project-Butler-Did-It-?-aka-Agatha-Christie-Wrote-It-?


# Butler Did it? Agatha Christie Wrote It? 
# Ask my NLP Neural Network

## Summary:
My project is dedicated to the classification problem: here I predict the author of a detective novel by the given paragraph. The following document will introduce the dataset and models I used, my challenges and key findings. This is a learning project, so it has not been designed for the direct practical application. However, the conclusions can be used for other classification problems as well.


## Description
This is a study project dedicated to a classification problem: as a result of the modelling I predict the author of the detective novel by given paragraph. The following document will introduce the dataset and models used, challenges and key findings. 

Once this is a training project, it has not been designed for practical application. However,  the conclusions regarding optimal number of classes or most powerful language-related features can be used for other classification problems as well. 

Alternatively, if you have found a mysterious manuscript in the attic and were a bit curious as to whether you were about to publish an unknown novel by a famous writer, this project may have a practical application too. :)

## Table of Contents:
Datasets
Feature engineering
Iteration 1. EDA
Iteration 2. Modelling
Iteration 2. EDA
Iteration 2. Modelling
Conclusions
 
 
## Datasets

For this project I created several datasets. The initial dataset has been collected with web scraping and initially contained a set of paragraphs from the popular detective novels and their authors. 

This initial dataset contained 6800 datapoints, including 31934  predictors, such as:
- part of the speech tags,
- sentiment assessment,
- length of the paragraphs,
- count of the stemmed words used by the author I have set n-grams length to (1,3) max_df to 3% of all corpus due to the heavy computations).

Then, after modelling described in the chapter “Iteration 1. Modelling”, I have made a decision to add more authors and paragraphs for each of them, exclude any paragraphs containing foreign words a' nd add another features, such as:
 - ratio of unique words given paragraph length,
- the patterns of use parts of speech repetitively (e.g. 3 adjectives in a row).

The updated dataset had finally had 45184 data points and 6271 predictors (I have set n-grams length to (1,3) and max_df to 0.3% of all corpus due to the heavy computations).

The results of the further modelling are to be described in the chapter “Iteration 2. Modelling”.

## Feature engineering

My initial hypothesis was that the collected features should represent the individual style of each author, as the dataset was designed to reflect diversity of authors from different epochs, countries and genders and also the books written in English originally or translated.

My guess was that these different authors had tendencies to use certain stylistic patterns, such as few or many adjectives and adverbs.
Both iterations of feature engineering (chapters “Iteration 1. EDA” and “Iteration 2. EDA” respectively) were dedicated to finding predictors inside the given text: I was designing features based on the information hidden in the paragraph itself, no ‘external’ data (such as publishing house or year of writing) were included.

Also, for the purpose of ‘pure’ analysis, I have excluded digits (that may point to the year when the story took place) and any personal or geographical names.

## Iteration 1. EDA

To be described
 
## Iteration 1. Modelling

In order to find an optimal model, I have tested several models with different numbers of classes (authors). 

![The table comparing models and score with number of classes from 2 to 10]
(https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Comparison%20across%20models_10%20authors.png)

 
While all the models worked efficiently with the number of classes equal to 1 or two, with an increase in the number of classes I have observed a significant drop down in accuracy of the model. 

At this point it became obvious that German words that had not been excluded from the text were power predictors (based on the feature importance reports for the Rain Forest Classifier and Support Vector Machine (ovr) model). Therefore, in the second iteration of modelling I have dropped the rows containing foreign words.

![RainForest Classifier: feature importance]
(https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/RFC_feature%20importance.png)

![RSupport Vector Machine: feature importance]
(https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/SVM_feature%20importance.png)

I have made a conclusion that the optimal balance was between 6 classes and Support Vector Machine model (svm.SVC(kernel='linear', decision_function_shape='ovr', C=1). 
This combination allowed to get the following scores:
- Accuracy score (train set) =0.998
- Accuracy score (test set) = 0.998

However, on the full dataset (34 classes) the results were sizeable worse:
- Accuracy score (train set) = 0.254
- Accuracy score (test set) = 0.473

My next hypothesis was that the score could be improved if I were to build a Neural Network, so that not only the power of predictors could contribute to the accuracy level, but also the known mistakes. So I have tested it on 7 classes with MLPClassifier(hidden_layer_sizes=(100,100,100,100,100,100), max_iter=1000,warm_start=True, random_state=42, activation='logistic')
However, it has not proven itself:
- Accuracy score (train set) = 1.0 (sic!)
- Accuracy score (test set) = 0.322

The confusion matrix shows that the majority of authors are more often confused with others rather than being identified correctly. 

This type of error in multi-classification has been well described by Maya R. Gupta,Samy Bengio (Google Inc. ) in the study ‘Training Highly Multiclass Classifiers’: ‘In practice, the more classes considered, the greater the chance that some classes will be easy to separate, but that some classes will be highly confusable.’ 

Despite the fact my dataset only has several dozen classes, not thousands of them, the issue is the same. So I will try to tackle this issue with a bigger dataset and new added features. The results are to be described in the chapter ‘Iteration 2. Modelling’.
