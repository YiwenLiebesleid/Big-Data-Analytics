# Online Sampling and Estimation via YouTube API
### Proposal
YouTube is the most popular video website in the world, and we have the estimation that there're approximately 500 million videos on YouTube. Then what about the total number of YouTube channels? I this experiment I sampled and estimated the number of YouTube channels using YouTube API.

### File description:
Here are some codes I wrote during the project, some of them are not useful though. I use a star(*) to distinguish the most important files.

### subset:
Sample channel ids via YouTube API to create a subset.
(This file is no longer used in later works)

### validate:
Find the pattern of id space.
Generate dataset manually in the same way of YouTube channels, in order to know if the estimator is valid.
The created dataset is data1.txt, I attached this file along with codes.

### (*)toyestimator
The real validating process is in this file.
Input: data1.txt
Parameters: m as sample times, L as prefix length.
Output: estimated value of total number of ids in subset, and the RMSE value.

### (*)estimator
Estimating process for real YouTube channel ids.
Input: DEVELOPER_KEYS (the most annoying part!)
If m<=80, then only need DEVELOPER_KEY2, otherwise need both DEVELOPER_KEY1 and DEVELOPER_KEY2.
Parameters: m as sample times, L as prefix length.
Output: estimated value of total number of ids in YouTube.

### CV:
Simple cross validation.
To show the number of returns for a particular prefix of length 2 follows Binomial distribution.
It returns the number of matching ids returned by search for every query.
