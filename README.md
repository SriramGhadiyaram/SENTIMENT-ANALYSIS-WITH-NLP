# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : CHINTHAPARTHI THISHITHA

*INTERN ID* : CT06DM1408

*DOMAIN* : MACHINE LEARNING

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

This project is a comprehensive implementation of a Sentiment Analysis system aimed at classifying textual data, specifically tweets, as either positive or negative. The primary goal is to apply machine learning techniques to understand the emotional tone of social media content. The entire process involves several crucial steps: importing necessary libraries, loading the dataset, preprocessing the text, vectorizing the data, building a machine learning model, and finally evaluating and visualizing the results.

To begin with, essential Python libraries are imported to handle different aspects of the project. These include pandas and numpy for data manipulation and numerical operations, matplotlib.pyplot and seaborn for data visualization, and re and string for text preprocessing. From the scikit-learn library, modules for splitting datasets, vectorizing text (TF-IDF), building a logistic regression model, and evaluating performance are also imported.

The dataset is loaded from an external CSV file containing tweets labeled with sentiments. Only the relevant columns—tweets and their corresponding labels—are selected. These columns are renamed to 'review' (the tweet text) and 'sentiment' (the label: 0 for negative, 1 for positive). A preview of the dataset is printed to ensure it has been loaded correctly and to get an idea of the structure.

Next, the raw text data is cleaned using a custom preprocess_text() function. Text data is inherently noisy, especially in social media contexts, so preprocessing is vital for improving model accuracy. This function converts all text to lowercase to maintain uniformity and then removes URLs, mentions (e.g., @username), hashtags, punctuation, numbers, and unnecessary white spaces. The cleaned text is stored in a new column called 'clean_review'. This step ensures that the text data is standardized and that irrelevant or misleading content is removed.

Once the data is cleaned, it is split into training and testing subsets using the train_test_split() function. This allows the model to be trained on one portion of the data and evaluated on another, ensuring that it generalizes well to new, unseen data. An 80/20 split is typically used, where 80% of the data is used for training and 20% for testing.

The text data is then transformed into numerical format using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. TF-IDF is a statistical method that reflects how important a word is to a document relative to the corpus. By limiting the vectorizer to the top 5000 features, the model can focus on the most relevant words while reducing computational complexity. This transformation converts the text data into a matrix of numbers that can be fed into a machine learning algorithm.

A Logistic Regression model is then trained on the vectorized data. Logistic Regression is a widely used algorithm for binary classification problems, making it well-suited for sentiment analysis. It learns the relationship between the features (words) and the labels (sentiments) and builds a model that can predict the sentiment of new, unseen text.

After training, the model's performance is evaluated using several metrics. The accuracy score gives a basic idea of how many predictions were correct. The classification report provides more detailed information, including precision, recall, and F1-score for both positive and negative sentiments. These metrics help assess the model's performance from multiple angles. Additionally, a confusion matrix is created and visualized using a heatmap. This visual tool helps in identifying where the model is performing well and where it is making mistakes—such as confusing positive tweets for negative ones or vice versa.

In conclusion, this sentiment analysis project effectively demonstrates how machine learning can be used to analyze and classify text data. Starting from raw tweets, it walks through the entire process of cleaning, transforming, modeling, and evaluating the data. The use of Logistic Regression and TF-IDF ensures the approach remains efficient and interpretable. The modular design also makes it easy to extend—additional steps like stopword removal, lemmatization, or using advanced models like SVM or deep neural networks can be incorporated to further enhance performance. This project serves as a solid foundation for anyone looking to explore Natural Language Processing (NLP) and text classification in real-world applications.

