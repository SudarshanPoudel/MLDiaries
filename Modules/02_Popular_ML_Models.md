# Decision Tree
A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It is a non-parametric model that makes decisions based on the features of the input data.
Let's understand it with simple classification example.

![Decision Tree](../Plots/Module_2/dt_spam.png)

As shown in the figure, It's a very simple spam classifier decision tree. To classify any mail, we'll first look if it contain word "OFFER", if no model conclude that as "NOT SPAM". Otherwise we'll look another feature "No of links", if it's more then 5 model classify mail as "SPAM" else it'll classify as "NOT SPAM".
<br>
So basically we split internal nodes into 2 or more child nodes based on certain feature's criteria, and when we reach the leaf node we'll have your output. Now with the data we have we need our model to learn which features to look initially and which features later on as well as the criteria of splitting. A model learns to make a decision tree through a process called recursive partitioning, which we'll discuss after Impurity metrics.

# Impurity Metrics
Impurity metrics are used in decision tree algorithms to measure the homogeneity of the labels at a node. If one node contains all data of only class 'A' then that node is fully pure, while having lat's say data of class 'A' and 'B' in ratio of 50-50 is highly impure. The goal of a decision tree is to create nodes that are as pure as possible, i.e. we'll try to split any node such that impurity in it's child class is as low as possible.
Two of the most commonly used impurity Metrics are as follows:

## 1. Gini Impurity:
Gini impurity measures the likelihood of a random sample being incorrectly classified based on the distribution of class labels in a node. For e.g. let's assume node as a bucket and data sample as red and blue balls. If bucket is mostly filled with red balls and a few blue balls, the Gini impurity would be low because if we randomly pick a ball, it's likely to be red. But if the bucket has an equal number of red and blue balls, the Gini impurity would be higher because picking a ball randomly could result in either color. 
We can calculate Gini impurity with formula:

<p>
$$
 Gini(p) = 1 - \sum_{i=1}^{K} (p_i)^2 
$$
where $p_i$ is the proportion of samples that belong to class $ i $ in node $p$.
</p>

## 2. Entropy:
Entropy is a measure of the impurity or uncertainty in a dataset. It quantifies how much information is needed to describe the randomness or disorder of a set of data points with respect to their class labels. A low entropy indicates a more ordered or homogeneous set, while a high entropy signifies greater disorder or diversity. ALthough It's slightly more computationally expensive due to logarithmic calculations, it's more sensitive nature to changes in the class probabilities, often leads to more balanced trees.
<p>
The entropy $ H(S) $ of a set $ S $ with $ K $ different classes is calculated as:
   
   $$ H(S) = - \sum_{i=1}^{K} p_i   \log_2(p_i) $$
   
   where $ p_i $ is the proportion of samples belonging to class $ i $ in the set $ S $.
</p>

# Recursive Partitioning

# Overfitting In Decision Tree

# Regularization : Early Stopping

# Regularization : Pruning

# Support Vector Machines (SVM)

# K-Nearest Neighbors

# KD Tree

# K-NN for Classification & Regression


In the context of decision trees, **entropy** is a measure of the impurity or uncertainty in a dataset. It quantifies how much information is needed to describe the randomness or disorder of a set of data points with respect to their class labels.
