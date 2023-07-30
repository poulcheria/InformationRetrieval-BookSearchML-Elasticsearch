# InformationRetrieval-BookSearchML-Elasticsearch
This repository showcases the implementation of a book search engine based on Elasticsearch, which utilizes advanced machine learning techniques like word embeddings, neural networks, and k-means clustering for result ranking. The project is developed in Python as part of a university assignment for the Computer Engineering and Informatics Department (CEID).  

## Overview  
1. **csvPass.py, csvRatingPass.py, csvUsersPass.py**: These Python scripts read records from CSV files and import them into Elasticsearch.
2. **elasticsearch_import.py**: This script takes an alphanumeric input as a search query and returns a list of books that match the query. The results are presented in descending order based on the predefined Elasticsearch similarity metric. This ranking allows users to find the most relevant books that align with their search query.
3. **neural_network.py**:The neural network in this project is designed to predict user ratings based on book summaries using the Word2Vec model and Keras.
4. **k_means**: The k-means program incorporates the wordembeddings function from **neural_network.py** to analyze a set of summaries. The resulting vectors obtained from the word embeddings process serve as input for the k-means clustering algorithm. In this implementation, the k-means algorithm is modified to utilize the **cosine similarity** as a distance metric.

## Neural Network for User Ratings Prediction
1. **Word Embeddings using Word2Vec**  
The wordembeddings function tokenizes, preprocesses, and filters the words in each book summary. It then employs the Word2Vec model from gensim to convert the words into 10-dimensional decimal vectors. The vectors representing each summary are summed by axes, resulting in a final decimal vector representation for each summary.
2. **Neural Network Architecture**  
The neural network architecture is constructed using Keras. It consists of three layers:  

- Input Layer: The input layer has 10 units, corresponding to the 10-dimensional decimal vector representations of the book summaries.
- Hidden Layer: The hidden layer has 32 units with the ReLU activation function, which introduces non-linearity to the model, allowing it to learn complex patterns in the data.
- Output Layer: The output layer has 1 unit with the linear activation function, as this is a regression problem, and we are predicting continuous values (user ratings).
3. **Model Compilation**
The Keras model is compiled using the mean squared error (MSE) loss function and the Adam optimizer.
4. **Model Training**
The neural network is trained on the dataset, which includes the 10-dimensional vector representations of the book summaries as inputs and the corresponding user ratings as outputs. The training is performed for 1000 epochs, with a batch size of 16.
5. **Model Evaluation**
The model is evaluated on the training dataset to assess its accuracy. Additionally, it is used to make predictions on the test dataset to predict user ratings for unseen data.
6. **Predicting User Ratings for Unrated Books**
The model is also used to predict user ratings for books that have not been rated yet. The book summaries for these unrated books are passed through the wordembeddings function, and the resulting vectors are used as inputs to the neural network for rating predictions.

## K-means Clustering to Analyze a Dataset of Book Summaries. 
1. **K-means Clustering**
After obtaining the word embeddings for all book summaries, the k-means algorithm is applied to group the summaries into 10 clusters based on their vector representations. The k-means algorithm uses the cosine similarity as a distance metric for clustering.
2. **Centroids of Clusters**
The centroids of the 10 clusters are determined after the k-means clustering process. Each centroid represents a cluster's central point and characterizes the average features of the summaries within that cluster.

3. **Cluster Analysis**
For each of the 10 clusters, an analysis is performed to extract meaningful information:  

  - Average Age : The average age of users who belong to that cluster. 
  - Most Frequent Location: The most frequently occurring location of users in the cluster.
  - Average Rating: The average rating of books provided by users in the cluster. 
  - Most Frequent Category: The most frequently occurring category of books in the cluster.  
The analysis provides insights into the characteristics and preferences of users in each cluster, allowing for targeted recommendations and personalization.

## Technologies Used
  -Python
  -Elasticsearch
  -pandas
  -gensim
  -nltk
  -tensorflow
  -keras
  -scikit-learn 

## Installation and Usage
1. Install the required Python libraries using pip:
`pip install pandas gensim nltk tensorflow keras scikit-learn`  
2. Make sure you have Elasticsearch installed on your system.
3. Clone this repository to your local machine.
4. Due to the large size of the dataset, the data folder contains a compressed zip file. Download the zip file from the repository and extract it locally. The data files required for the book search engine will be available after extraction.
5. To run the book search engine and obtain book recommendations, execute the following Python scripts:  
  csvPass.py, csvRatingPass.py, csvUsersPass.py, elastocsearch_import.py
6. For further analysis using word embeddings, neural networks, and k-means clustering, refer to the respective scripts neural_network.py and k_means.py. Modify the input parameters as needed.

**_Note:_**: It is recommended to consider changing the word embeddings dimensions from 10 to a much bigger value, at least 10 times more, for improved performance and better representation of word semantics in the book search engine. Experiment with different dimensions to find the optimal setting that suits your specific use case.
