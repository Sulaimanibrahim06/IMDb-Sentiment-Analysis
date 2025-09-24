# IMDb-Sentiment-Analysis

## Project Overveiw:

In this project I have trained sentiment analysis models on the IMDb dataset
These models predict if the given review about the movie is either positive or negative

I have compared the 5 models that I have trained on the basis of Accuracy,Precision,Recall and F1 Score

## Approach:

### Data Preprocessing:

-Text Cleaning:

Raw data always has some noise and unwanted stuff such as HTML Tags,punctuations etc.. so we removed these using a pipeline that 

1.Converts text to lowercase.

2.Removes HTML tags such as br.

3.Removes punctuation, numbers, and special characters.

4.Removes common stopwords to focus on important terms.

-Vectorization:

We know that the Machine Learning model can only understand numbers and not text, hence to solve that I have applied TF-IDF Vectorization which removes common words like "this","is" etc..

## Models Implemented:

### Logistic Regression:

Logistic Regression is a linear model that is widely used for binary and multi-class classification problems. It models the probability that a given input belongs to a particular class using the logistic (sigmoid) function.
Reason To use this model:
Simplicity: Easy to implement and interpret.
Speed: Very fast to train even for high-dimensional datasets.
Baseline: Often used as a baseline to compare more complex models.
Probabilistic Output: Provides probability estimates for class predictions, useful for ROC curves and decision thresholds.
Logistic Regression is a baseline model to understand the initial performance on our dataset. It served as a reference point to compare with more complex models like SVM and Random Forest.

<img width="515" height="455" alt="image" src="https://github.com/user-attachments/assets/9844493e-7e49-4932-ac83-0a2c1be0cb3c" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/82b099a6-65d0-4fbb-a272-08b68dda53ae" />
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/bcdb6b93-7a97-48a2-b08b-418a5c5962f0" />

### Support Vector Machine (SVM):

Support Vector Machines are margin-based classifiers that aim to find a decision boundary (hyperplane) that best separates classes with the largest margin.
Reason To use this model:
High-dimensional capability: Works well with sparse TF-IDF vectors in text classification.
Robustness: Effective in cases where classes are not perfectly separable.
Flexibility: Can handle linear and non-linear problems via kernels.
We used SVM to compare performance against Logistic Regression and to leverage its ability to handle high-dimensional sparse data produced by TF-IDF vectorization.

<img width="515" height="455" alt="image" src="https://github.com/user-attachments/assets/4231d3ee-3ca1-4a71-b5c5-31510e1d5502" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/a1e11d28-7130-4a5d-8bab-e947674820b1" />
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/a5d33d93-8dae-4326-afec-d2baff38daa3" />

### Naive Bayes (NB):

Naive Bayes is a probabilistic classifier based on Bayes’ theorem,For text classification, the Multinomial Naive Bayes variant is commonly used since it handles word occurrence counts effectively.
Reason to Use this Model:
Extremely fast to train and predict.
Performs well for text classification, especially with bag-of-words or TF-IDF representations.
Works well with small datasets.
Baseline model for text classification tasks.
Naive Bayes was implemented as a probabilistic baseline to evaluate against more computationally expensive models. Its speed made it a useful quick benchmark for comparing accuracy and other metrics.

<img width="515" height="455" alt="image" src="https://github.com/user-attachments/assets/eabaa7a0-fd58-425d-950b-4b9436e371fc" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/2777c5d9-e0e5-4f33-8a2f-16386e02c793" />
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/0db37540-fe0f-44ea-b62e-488e64c00700" />

### Random Forest:

Random Forest is an ensemble learning method that constructs multiple decision trees and combines their predictions. It uses bagging (bootstrap aggregating) and feature randomness to create decorrelated trees whose averaged prediction improves generalization.
Reason to Use this Model:
Non-linear capability: Captures complex relationships between features.
Robustness: Reduces overfitting through ensemble averaging.
Versatility: Works well on structured and high-dimensional data.
Feature importance: Can give insights into feature significance.
Random Forest was chosen as a non-linear model benchmark to compare with linear and probabilistic classifiers. It provided insight into whether complex relationships in the data could improve prediction performance beyond simpler models.

<img width="515" height="455" alt="image" src="https://github.com/user-attachments/assets/a9e2fad2-a5fd-49ba-8f6e-3646c83d1f44" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/b44b5b01-846a-4e82-ad4f-b09944161cf7" />
<img width="584" height="455" alt="image" src="https://github.com/user-attachments/assets/1110baf9-981e-4248-a47e-19f67e3f59c4" />

### LSTM:

While Logistic Regression, SVM, Naive Bayes, and Random Forest are powerful models for classical machine learning, they rely on feature engineering (such as TF-IDF) and cannot directly capture sequential dependencies in textual data.

The LSTM implementation follows a clean and efficient pipeline:

a. Tokenization

Text is first converted into numerical sequences using a tokenizer.
This maps each unique word in the dataset to a unique integer index.

b. Padding

Sequences are padded to the same length to allow batch processing.

c. Embedding Layer

An Embedding layer converts integer word indices into dense vectors.
These vectors represent semantic meaning in a continuous space.

d. LSTM Layer

The core LSTM layer learns sequential patterns from the embedded input, keeping track of dependencies and context in the sequence.

e. Dense Layers

After the LSTM output, fully connected (Dense) layers transform the learned representation to a binary classification output.

f. Output Layer

A single neuron with a sigmoid activation gives the probability of the positive class.

Reason to Use this model:

Captures word order and context: Unlike TF-IDF with classical ML, LSTM inherently considers the sequence of words.

Deep learning benchmark: Allows comparison between classical ML models and a deep learning approach.

Exploration of advanced techniques: Helps expand the project scope for research or academic purposes.

Performance testing: Enables testing whether sequential modeling yields better accuracy and metrics compared to bag-of-words approaches.

<img width="522" height="470" alt="image" src="https://github.com/user-attachments/assets/b21e87b7-3a5d-4457-ac5b-524d8404b7a1" />
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/4449df05-f9f5-45d5-b077-0aea53ec7506" />



## Results:

I compared the 5 models that I trained using the 4 parameters (Accuracy,Percision,Recall,F1-Score)

📊 Model Comparison:

| Model                  | Logistic Regression | Support Vector Machine | Naive Bayes  | Random Forest | LSTM |
| ---------------------- | -------- | --------- | ------ | -------- | -------- |
| Accuracy    | 0.889800 | 0.881200   |    0.856800   |  0.853100 | 0.866800     |
| Precision | 0.883661 | 0.876085   |    0.865424  |   0.848019 | 0.869014     |
| Recall            | 0.897800 | 0.888000  |     0.845000   |  0.860400 | 0.863800     |
| F1-Score          | 0.890675 | 0.882002  |     0.855090  |   0.854165  |     NaN     |

<img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/b2a5b0fd-fe7b-4c0e-9fa2-0e669088b8e8" />
















