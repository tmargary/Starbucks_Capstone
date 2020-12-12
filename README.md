# Starbucks Capstone Project

## `0` Project Overview

The goal of the project is to build a user-based recommendation system that identifies users that are similar to each other and makes recommendations, excluding the offers that have already been received or viewed without an interaction.

To achieve this, I will be using singular value decomposition (SVD), the most common method in collaborative filtering where it relies on te past user-item data. In case of missing recommendations for certain users, I will find similar users based on their dot product and recommend the offers accordingly. The expected solution is to recommend at least one offer to the 1700 usrers that we have in the dataset, as well as evaluate how well the model works.

Note: I have assumed that in case the user has accepted all the 10 offers, the customer is a loyal customer and does not need any offers in the short to medium term. On the other hand, if the customer has viewed all of the offers without receiving them, there is no need to continue the recommendations in the short to medium term.

## Data Cleaning
The Starbucks Capstone Challenge by Udacity was a great challenge to develop data cleaning skills. The challenge provides three datasets:<br/>
- profile.json<br/>
Rewards program users (17000 users x 5 fields)<br/>
- portfolio.json<br/>
Offers sent during the 30-day test period (10 offers x 6 fields)<br/>
- transcript.json<br/>
Event log (306648 events x 4 fields)<br/>

The first thing I have done was mapping the cyphered ID's (`744d603ef08c4f33af5a61c8c7628d1c`) data with numbers (`123`). Next, after merging the tables, I have generated additional columns for channels of Starbucks marketing and calculated the membership length.

## `1` Data Cleaning

I have started the cleaning process by mapping the cyphered ID's (`744d603ef08c4f33af5a61c8c7628d1c`) data with numbers (`123`). I have done this by the `column_mapper` function below. 

The next thing that I have done was to separate the `value` column in the `transcript` dataset. Initially, the column contained values, such as:


| value |
| --- |
|{'offer id': '9b98b8c7a33c4b65b9aebfe6a799e6d9'}|
|{'amount': 4.89}|
|{'amount': 4.23, 'offer id': 'f19421c1d4aa40978ebb69ca19b0e20d'}|
|...|

I have broken down these values based on their key-value pairs and created a new column for each key. Next, I have solved the anomaly of having `offer id` and `offer_id` as different keys. I made sure that these values are included in the same column.

After merging all three semi-clean datasets together, I have generated additional columns for the marketing channels and calculated the membership length as a new column.

An interesting thing that I have learned from this part of the cleaning was that `np.nan` is not the same as the pandas' `nan`. So I had to use something like this: `pd.isna(x[column]) == pd.isna(np.nan)` for making a comparison in one of the lambda functions below.

## EDA

![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/age.png)
![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/offer_type.png)

## SVD
After cleaning the dataset and exploring some details, I have built user matrix factorization to make offer recommendations to the users. I have used Singular Value Decomposition from numpy on the user-item matrix: u, s, vt = np.linalg.svd(order_received_mat).</br>
In order to minimize the prediction error, we have to choose the number of latent features.</br></br>
![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/Screenshot_6.png)</br>
As it is obvious from the graph, when `k` equals 10, we have the least amount of error (`0.0`).</br>

The final output of the model is also a list that contains the offer ID's that should be recommended to the user. The results look like this: "`The user 14530 will get the following offers: [5.0, 7.0, 10.0]`".</br>

## Model Evaluation and Validation
According to Udacity, "from the above, we can't really be sure how many features to use, because simply having a better way to predict the 1's and 0's of the matrix doesn't exactly give us an indication of if we are able to make good recommendations. Instead, we might split our dataset into a training and test set of data."</br>

At this point, we have two users that do not have recommendations yet. To make further recommendations for them, I have found similar users to them using the dot product between users. While building the functions, I have assumed that in case the user has accepted all the 10 offers, the customer is a loyal customer and does not need any offers in the short to medium term. On the other hand, if the customer has viewed all of the offers without receiving them, there is no need to continue the recommendations in the short to medium term.</br>

In order to make the recommendations, I have created a user_item matrix and found similar_users for each user. Then I got the recommendations and filtered out already accepted and not received offers.</br>

I have created the user-offer matrix with 1's and 0's which I will use later for finding similar users.</br>

The result of similar users is a list, such as "`The 3 most similar users to user 14530 are: [1, 8, 11]`."</br>

## Conclusion

This is a good start that can be used for creating a recommendation engine and integrating it in the Starbucks app. Since the dataset includes only 10 offers, it would be interesting to see how the model behaves when a larger dataset is provided.</br>

Using User-User Based Collaborative Filtering, I have managed to make recommendations for almost all the customers in my test data frame. Next, I have found out that I still have two users that need recommendations. To solve the issue, I have found similar users. As a result, the model covers all the 17000 customers in the dataset.</br>

To make the recommendations more relevant to the users, we might as well consider more demographic data about the customers to build a more complex engine. Also, it is also important to consider factors, such as relevance, novelty, serendipity, increased diversity to increase the quality of the recommendations.</br>

A logical next step should be A/B testing to see how successful the model behaves in reality.</br>

## Resources
- **Python Version:** 3.8<br/>
- **Packages:** pandas, numpy, json, math, seaborn </br>
- **Data Source:** Starbucks (by Udacity's Udacity Data Scientist Nanodegree Program)</br>
- **Code Source for SVD:** Udacity's Recommendations_with_IBM
