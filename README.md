# Starbucks Capstone Project

## Project Overview

I have combined transaction, demographic, and offer data to make a recommendation system for Starbucks customers. The goal of the project is to build a user-based recommendation system that identifies users that are similar to the queried user and makes recommendations, excluding the offers that have already been received or viewed without an interaction. I have assumed that in case the user has accepted all the 10 offers, the customer is a loyal customer and does not need any offers in the short to medium term. On the other hand, if the customer has viewed all of the offers without receiving them, there is no need to continue the recommendations in the short to medium term.

## Data Cleaning
The Starbucks Capstone Challenge by Udacity was a great challenge to develop data cleaning skills. The challenge provides three datasets:<br/>
- profile.json<br/>
Rewards program users (17000 users x 5 fields)<br/>
- portfolio.json<br/>
Offers sent during the 30-day test period (10 offers x 6 fields)<br/>
- transcript.json<br/>
Event log (306648 events x 4 fields)<br/>

The first thing I have done was mapping the cyphered ID's (`744d603ef08c4f33af5a61c8c7628d1c`) data with numbers (`123`). Next, after merging the tables, I have generated additional columns for channels of Starbucks marketing and calculated the membership length.

## EDA

![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/age.png)
![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/offer_type.png)

## SVD
After cleaning the dataset and exploring some details, I have built user matrix factorization to make offer recommendations to the users. I have used Singular Value Decomposition from numpy on the user-item matrix: u, s, vt = np.linalg.svd(order_received_mat).</br>
In order to minimize the prediction error, we have to choose the number of latent features.</br></br>
![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/Screenshot_6.png)</br>
As it is obvious from the graph, when `k` equals 10, we have the least amount of error (`0.0`).

## Model Evaluation and Validation
According to Udacity, "from the above, we can't really be sure how many features to use, because simply having a better way to predict the 1's and 0's of the matrix doesn't exactly give us an indication of if we are able to make good recommendations. Instead, we might split our dataset into a training and test set of data."</br>

At this point, we have two users that do not have recommendations yet. To make further recommendations for them, I have found similar users to them using the dot product between users. While building the functions, I have assumed that in case the user has accepted all the 10 offers, the customer is a loyal customer and does not need any offers in the short to medium term. On the other hand, if the customer has viewed all of the offers without receiving them, there is no need to continue the recommendations in the short to medium term.</br>

In order to make the recommendations, I have created a user_item matrix and found similar_users for each user. Then I got the recommendations and filtered out already accepted and not received offers.</br>

I have created the user-offer matrix with 1's and 0's which I will use later for finding similar users.</br>

The result of similar users is a list, such as "`The 3 most similar users to user 14530 are: [1, 8, 11]`."

## Resources
- **Python Version:** 3.8<br/>
- **Packages:** pandas, numpy, json, math, seaborn </br>
- **Data Source:** Starbucks (by Udacity's Udacity Data Scientist Nanodegree Program)</br>
- **Code Source for SVD:** Udacity's Recommendations_with_IBM
