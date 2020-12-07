# Starbucks Capstone Project

## Project Overview

Combined transaction, demographic and offer data to make a recommendation system for Starbucks customers. The user-based recommendation system identifies users that are similar to the queried user and makes recommendations, excluding the offers that have already been received or vewied without an interuction. I have assumed that in case the user has accepted all the 10 offers, the customer is a loyal customer and does not need any offers in the short to medium term. On the other hand, if the customer has viewed all of the offers without receiving them, there is no need to continue the recommendations in the short to medium term.

## EDA

![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/age.png | width=50)
![](https://github.com/tmargary/Starbucks_Capstone/blob/main/graphs/offer_type.png)

## Metrics

I have calculated the number of users that I can make reccommendations for with the abovementioned approach, and 16994 out of 17000 will be receiving recommendations.

## Funk SVD

In order to make the recommendations, I have created a `user_item` matrix and found `similar_users` for each user. Then I got the recommendations and filtered out `already accepted` and  `not received` offers.

## Resources
Mentorship: Udacity team. Special thanks to Rajat P.</br>
Python Version: 3.8</br>
Packages: pandas, numpy, json, math, seaborn </br>
Data Source: Figure Eight</br>
