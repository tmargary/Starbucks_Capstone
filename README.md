# Starbucks Capstone Project

## `0` Project Overview

The goal of the project is to build a user-based recommendation system that identifies users that are similar to each other and makes recommendations, excluding the offers that have already been received or viewed without an interaction.

To achieve this, I will be using singular value decomposition (SVD), the most common method in collaborative filtering where it relies on te past user-item data. In case of missing recommendations for certain users, I will find similar users based on their dot product and recommend the offers accordingly. The expected solution is to recommend at least one offer to the 1700 usrers that we have in the dataset, as well as evaluate how well the model works.

Note: I have assumed that in case the user has accepted all the 10 offers, the customer is a loyal customer and does not need any offers in the short to medium term. On the other hand, if the customer has viewed all of the offers without receiving them, there is no need to continue the recommendations in the short to medium term.

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

## `2` EDA

![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/age.png)
![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/offer_type.png)

## `3` User-User Based Collaborative Filtering

`3.1` After cleaning the dataset and exploring some details, I will build user matrix factorization to make offer recommendations to the users.

I will be using SVD, a traditional approach to matrix factorization. To perform SVD, I will be creating a **user_item** matrix and split it into three matrices:

$$ U \Sigma V^T $$

I will use `numpy`'s `linalg.svd()` function to get these components. 
It has the following dimensions:

$$ U_{n x k} $$

$$\Sigma_{k x k} $$

$$V^T_{k x m} $$

where:

1. n is the number of Starbucks customers
2. k is the number of latent features to keep (10 for this case)
3. m is the number of offers

I have chosen SVD as an algorithm behind my recommender system because it can incorporate implicit information that’s not directly given in the dataset but can be derived by looking at the offers frequently received or completed. Using this capability we can estimate wheather a user is going to accept the offer. If the prediction is 1, we should send the offer. Compared to other methods, SVD is especially useful because it's easy to explain how it works and general audience of business executives can easily understand the gist of how the technique works.

**Metric to measure performance**: To minimize the prediction error, I have to choose the number of latent features and calculate the error rate. As an essential metric, I will also look at the number of customers that are left without recommendations.

## `4` Refinement

In the earlier stages of the project, I was trying to do exploratory data analysis to find a deeper connection between the customers' demographics and the offers that they accept. After some attempts, I understood that the best approach to do this would be neural networks, and I did not intend to go with this path because the dataset might not be big enough for that approach. My goal was to deepen my knowledge in the recommender systems, and this was an excellent opportunity to try solving the Starbucks problem from that perspective.

Initially, I manually selected an arbitrary number of `k` (latent factors) to calculate the error. I wanted to keep the `k` as small as possible to make the solution more scalable for later use. After visualizing the error rate vs. `k`, I have chosen 10 as the most optimal number for the problem.

When I realized that I still have 2 users that do not have any recommendations, I have tried to recommend them the most popular offer in the dataset. After doing more EDA of the dataset, I came to the conclusion that the popularity of the offers does not vary significantly. So I have built another function that finds similar users and makes recommendations accordingly.

In real life, the number of offers is going to be more, and most probably, there are going to be some offers that are more popular than others, and having that information will be particularly helpful for brand new customers (also known as the cold start problem) who will join Starbucks in the future.

## `5` Final results

After cleaning the dataset and doing some exploratory data analysis, I have used SVD to make recommendations for the customers that we have in the dataset. The result of SVD was a matrix where the rows represent the cutomers, the columns represent the offers, and 1's and 0's indicate whether or not the offer should be extended. 

Since the matrix doesn't exactly give us an indication of if we are able to make good recommendations, 

So, eseentially we ended up learning the $ U \Sigma V^T $ components of the following matrix:

| 1 | 2 | 3 | ... | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | ... |  0 | 0 | 1 |
| 0 | 0 | 1 | ... |  0 | 0 | 0 |
|  0 | 0 | 0 | ... |  1 |  0 | 0 |
|  0 | 0 | 1 | ... |  1 |  0 | 0 |
|  1 |  1 |  0 | ... |  0 | 0 | 0 |
|  0 | 0 | 1 | ... |  0 | 1 |  0 |
| ... | ... | ... | ... | ... | ... | ... |

These components will later be used for making recommendations for the customer, but now to find out how well the model is doing so, I have split the dataset into training and test sets. I have calculated the number of users that are left out without any recommendation and by using the similarity technique I have managed to make similarity based recommendations to these users.

Using the learned matrices, now we can use the user_user_recs() function to get recommendations for a specific user. As a result of using the techniques that are detailed above, the recommendation model covers all the 17000 customers in the dataset.

One possible usage of the model in real world might be integrating the model with the Starbucks app, which will later predict potential offers for the user and recommend them. After a while of collecting more data, we can re-evaluate the performance of the model and make necessary adjustments.

## `6` Improvement

This is a good start that can be used for creating a recommendation engine and integrating it in the Starbucks app. To make the recommendations more relevant for the users, we might as well consider more demographic data about the customers to build a more complex engine. It is also essential to consider factors such as relevance, novelty, serendipity and increased diversity to increase the quality of the recommendations. Since the dataset includes only 10 offers, it would be interesting to see how the model behaves when a larger dataset is provided.

For a company like Starbucks, it's also crucial to have a scalable solution, and as such, the cleaning process of the dataset should be optimized, and the number of `k` latent factors might be reduced to decrease the computational complexity.  

A logical next step should be A/B testing to see how successful the model behaves in reality.

## `7` Resources
- **Python Version:** 3.8<br/>
- **Packages:** pandas, numpy, json, math, seaborn </br>
- **Data Source:** Starbucks (by Udacity's Udacity Data Scientist Nanodegree Program)</br>
- **Code Source for SVD:** Udacity's Recommendations_with_IBM
