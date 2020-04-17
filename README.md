# Butler Did it? Agatha Christie Wrote It? Ask NLP Neural Network

## Summary:
My project is dedicated to the classification problem: here I predict the author of a detective novel by the given paragraph. The following document will introduce the dataset and models I used, challenges and key findings. This is a learning project, so it has not been designed for the direct practical application. However, the conclusions may be used for other classification cases as well.

## Description and application
This project focuses on multi-class classification based on the Natural Language Processing: as a result of the modelling I predict the author of the detective novel by a paragraph. Once this is a training project, the aim is to practice NLP skills and compare different models, nethertheless, the conclusions like optimal number of classes or most powerful language-related features can be used for other classification problems as well. 

Alternatively, if the one had found a mysterious manuscript in the attic and would have been curious as to whether it had been an unknown novel by a famous writer, this project could have given a bit of practical advice too. :)

![](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/%D0%A5%D0%BE%D0%BB%D0%BC%D1%81_%D0%BE%D0%BD.jpg)

## Table of Contents:

- [Datasets](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/README.md#datasets)
- [Feature engineering](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/README.md#feature-engineering)
- [Iteration 1. EDA](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/README.md#iteration-1-eda)
- [Iteration 1. Modelling](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/README.md#iteration-1-modelling)
- [Iteration 2. EDA](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/README.md#iteration-2-eda)
- [Iteration 2. Modelling](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/README.md#iteration-2-modelling)
- [Conclusions](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/README.md#conclusions)
 
 
## Datasets

For this project I created several datasets. The [initial dataset](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Iteration%201/df_40_authors_all_features_03percent.csv.zip) has been collected with web scraping and contains a set of paragraphs from the popular detective novels by 40 authors. These 3200 datapoints include 1952  predictors, such as:
- part of the speech tags,
- sentiment assessment,
- length of the paragraphs,
- count of the stemmed words used by each author.

Based of the results of the modelling described in the chapter “Iteration 1. Modelling”, I have made a decision to add more authors and paragraphs, exclude any paragraphs containing foreign words and include several new features, such as:
- ratio of unique words given paragraph length,
- the patterns of use parts of speech repetitively (e.g. 3 adjectives in a row).

The updated dataset had finally had 45184 data points and 6271 predictors I still had to set n-grams length to (1,3) and min_df to 0.3% of all corpus due to the heavy computations.


## Feature engineering

Since my plan was to only use the features extracted from the text itself, I came up with the following list od features:
- Length of the paragraph
- Sentiment analysis metrics: number of positive, neutral and negative words, along with objectivity and positive_vs_negiveat score
- Parts of the speech and punctuation marks counts
- Most common tokens (top 0.3% of the corpus, n-grams length equal to (1,3) min_df)

In order to keep the dataset balanced I only chose paragraphs between 300 and 600 symbols length, which tuirned out to be a mistake. Also, for the purpose of ‘pure’ analysis, I have excluded digits (that may point to the year when the story took place) and any personal or geographical names.

## Iteration 1. EDA

My hypothesis was that the collected features should represent the individual style of each author, as the dataset was designed to reflect diversity of authors from different epochs, countries and genders and also the books written in English originally or translated.

Exploration of the data supported my theory, for example, different authors were proven to use more or less 'gruesome' vocabulary:

![Average positive versus negative score for each author](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/40_authors_pos_vs_neg_score.png)

At the same time, simple count of the most frequent words (Count Vectorization) showed that different authors had different most popular words, some of them also used different punctuation marks and parts of the speech more frequently than others.

![Top 50 words frequency across 40 authors](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/40_authors_top_50_words.png)

 
## Iteration 1. Modelling

In order to find an optimal model, I have tested several models with different numbers of classes (authors). 

![The table comparing train score of different models with number of classes from 2 to 10](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/Accuracy%20on%20the%20train%20set%20(40%20authors).png)

![The table comparing test score of different models with number of classes from 2 to 10](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/Accuracy%20on%20the%20test%20set%20(40%20authors).png)
 
Unfortunately, all the models showed low accuracy even with the number of classes equal to 2. Needless to say, with the increase of the number of classes I  observed a significant drop down in accuracy of the predictions. However, some of them worked better tha others.

![Dynamic of the accuracy worsening for different models](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/Drop%20in%20score%20of%20the%20diffrent%20models.png)

The relatively better results were achieved by Neural Networks: Perceptron and Multi-Layer Perceptron, which is can be explained by the ability of Neural Networks to take the previous error into consideration (this may also explain the score for Gradient Boosting, as the priority in sub-setting the data is given to hard-to-fit data).

The final run of the same models on the full dataset showed the further worsening of the score. The precision and recall scores highlight that the classes were easily mixed up.  

![Comparison of the models' score for 40 authors classsification](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/Scores%20on%20the%20full%20dataset.png)

The confusion matrices show that the majority of authors is more often confused with others rather than being identified correctly. 

![Fragment of Logistic Regression confusion matrix](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/LR_confusion_matrix.png)

So, according to Logistic Regression model, Agatha Christie or Arthur Conan Doyle can equally often be labelled as Arthur B. Reeve.

![Fragment of Random Forest Classifier confusion matrix](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/RFC_confusion%20matrix.png)

Random Forest Classifier labels Anna Catherine Green as Anthony Berkeley.

![Fragment of Multi-Layer Perceptron confusion matrix](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/MLP_confusion%20matrix.png)

The same issue occurs in every model. 

This type of error in multi-classification has been well described by Maya R. Gupta,Samy Bengio (Google Inc. ) in the study ‘Training Highly Multiclass Classifiers’: ‘In practice, the more classes considered, the greater the chance that some classes will be easy to separate, but that some classes will be highly confusable.’ 

Despite the fact my dataset only has 40 classes, unlike thousands of them described by Google researchers, the issue has been the same. So I in the next iteration my aim is to tackle this issue with a better Neural Network, bigger dataset and new added features. The results are to be described in the chapter ‘Iteration 2. Modelling’.

In order to find more powerful combination of features, I have compared top 20 predictors for Logistic Regression, Random Forest Classifier and Gradient Boosting Classifier.

![Logistic Regression feature_importance](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/LR_feature_importance_40%20authors.png)

![Random Forest Classifier feature_importance](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/RFC_feature_importance_40%20authors.png)

![Gradient Boosting Classifier feature_importance](https://github.com/TataAndBigData/NLP-capstone-project-Butler-Did-It-Agatha-Christie-Wrote-It-/blob/master/Illustrations/GB_feature_importance_40%20authors.png)

Apparently, the predictors associated with the frequency of the certain parts of speech or paragraph length have higher weights than the popular tokens. Therefore, in the second iteration of modelling I will not artificially balance length of the paragraphs and will add several features showing the use of parts of speech by different authors.

## Iteration 2. EDA

*To be described*

## Iteration 2. Modelling

*To be described*

## Conclusions

*To be described*
