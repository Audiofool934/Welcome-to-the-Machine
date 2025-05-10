# Machine / Statistical Learning: A Comprehensive Overview

- [Machine / Statistical Learning: A Comprehensive Overview](#machine--statistical-learning-a-comprehensive-overview)
  - [Introduction](#introduction)
  - [Types of Machine Learning](#types-of-machine-learning)
    - [By Learning Paradigm](#by-learning-paradigm)
      - [1. Supervised Learning](#1-supervised-learning)
      - [2. Unsupervised Learning](#2-unsupervised-learning)
      - [3. (Semi-supervised Learning)](#3-semi-supervised-learning)
      - [4. Reinforcement Learning (RL)](#4-reinforcement-learning-rl)
    - [By Models](#by-models)
      - [1. Probabilistic vs. Non-probabilistic (Deterministic) Models](#1-probabilistic-vs-non-probabilistic-deterministic-models)
      - [2. Linear vs. Non-linear Models](#2-linear-vs-non-linear-models)
      - [3. Parametric vs. Non-parametric Models](#3-parametric-vs-non-parametric-models)
    - [By Algorithms - Training Scheme](#by-algorithms---training-scheme)
      - [1. Online Learning](#1-online-learning)
      - [2. Batch Learning (Offline Learning)](#2-batch-learning-offline-learning)
    - [By Techniques - Underlying Methodologies](#by-techniques---underlying-methodologies)
      - [1. Bayesian Learning](#1-bayesian-learning)
      - [2. Kernel Methods](#2-kernel-methods)
  - [Representation, Inference, and Learning (Troika)](#representation-inference-and-learning-troika)
    - [1. Representation](#1-representation)
    - [2. Inference (Evaluation / Prediction)](#2-inference-evaluation--prediction)
    - [3. Learning (Optimization / Search)](#3-learning-optimization--search)
  - [Conclusion](#conclusion)


## Introduction

What is (Machine) Learning?

> Definition: A computer program is said to **learn** from experience $E$ with respect to some class of tasks $T$ and performance measure $P$, if its performance at tasks in $T$, as measured by $P$, improves with experience $E$.
-- Tom Mitchell, *Machine Learning(1997)*

Machine Learning (ML) is a subfield of artificial intelligence (AI) that focuses on designing systems that can learn from and make decisions or predictions based on data, without being explicitly programmed for each task. Statistical Learning (SL) is a closely related field, often considered a subfield of statistics, that shares many of the same goals and techniques. *While ML often emphasizes prediction accuracy and scalability, SL tends to place a stronger emphasis on model interpretability, uncertainty quantification, and statistical inference.* In practice, the terms are often used interchangeably 🤝, and the disciplines heavily borrow from each other.

---

## Types of Machine Learning

### By Learning Paradigm

This classification is based on the nature of the learning signal or feedback available to the learning system.

#### 1. Supervised Learning
   - **Definition:** The algorithm learns from a **labeled** dataset, meaning each data point (instance) is tagged with a correct output or label. The goal is to **learn a mapping function** that can predict the output for new, unseen inputs.
   - **Goal:** To approximate a function `f(X) = Y`, where `X` are input features and `Y` is the output label.
   - **Key Tasks:**
      - **Regression:** Predicting a continuous output value.
         - *Examples:* Predicting house prices, stock prices, temperature.
         - *Common Algorithms:* Linear Regression, Ridge Regression, Lasso Regression, Support Vector Regression (SVR), Decision Trees, Random Forests, Gradient Boosting Machines, Neural Networks.
      - **Classification:** Predicting a discrete class label (category).
         - *Examples:* Spam detection (spam/not spam), image recognition (cat/dog/car), medical diagnosis (disease/no disease).
         - *Common Algorithms:* Logistic Regression, k-Nearest Neighbors (k-NN), Support Vector Machines (SVM), Naive Bayes, Decision Trees, Random Forests, Gradient Boosting Machines, Neural Networks.
   - **Evaluation:** Typically involves metrics like Mean Squared Error (MSE) for regression, and Accuracy, Precision, Recall, F1-score, AUC-ROC for classification, usually on a held-out test set.

#### 2. Unsupervised Learning
   - **Definition:** The algorithm learns from an **unlabeled** dataset, meaning the data points have no predefined output labels. The goal is to discover hidden patterns, structures, or representations in the data.
   - **Goal:** To model the underlying structure or distribution in the data.
   - **Key Tasks:**
      - **Clustering:** Grouping similar data points together into clusters.
         - *Examples:* Customer segmentation, anomaly detection, grouping similar documents.
         - *Common Algorithms:* K-Means, Hierarchical Clustering, DBSCAN, Gaussian Mixture Models (GMM).
      - **Dimensionality Reduction:** Reducing the number of input features while preserving important information.
         - *Examples:* Feature extraction for visualization, noise reduction, improving performance of subsequent supervised learners.
         - *Common Algorithms:* Principal Component Analysis (PCA), t-distributed Stochastic Neighbor Embedding (t-SNE), Linear Discriminant Analysis (LDA - can also be supervised), Autoencoders.
      - **Association Rule Mining:** Discovering interesting relationships (rules) between variables in large datasets.
         - *Examples:* Market basket analysis ("customers who bought X also bought Y").
         - *Common Algorithms:* Apriori, Eclat, FP-Growth.
      - **Density Estimation:** Estimating the underlying probability density function of the data.
         - *Examples:* Anomaly detection, generating new data samples.
         - *Common Algorithms:* Kernel Density Estimation (KDE), Gaussian Mixture Models (GMM).

#### 3. (Semi-supervised Learning)
   - **Definition:** The algorithm learns from a dataset that contains a small amount of labeled data and a large amount of unlabeled data. It aims to leverage the unlabeled data to improve learning performance beyond what could be achieved with labeled data alone.
   - **Rationale:** Labeling data can be expensive and time-consuming.
   - **Approaches:**
      - **Self-training:** Train a model on labeled data, use it to predict labels for unlabeled data, add high-confidence predictions to the labeled set, and repeat.
      - **Co-training:** Use multiple models (views) of the data; each model labels data for the other.
      - **Generative Models:** Model the joint probability `P(X,Y)` and use it to infer labels.
      - **Graph-based Methods:** Represent data as a graph, propagate labels through the graph.
   - **Examples:** Using a large corpus of unlabeled text with a few labeled documents to improve text classification.

#### 4. Reinforcement Learning (RL)
   - **Definition:** An agent learns to make a sequence of decisions by interacting with an environment. The agent receives rewards or penalties based on its actions, and its goal is to learn a policy (a mapping from states to actions) that maximizes its cumulative reward over time.
   - **Key Components:**
      - **Agent:** The learner or decision-maker.
      - **Environment:** The external system the agent interacts with.
      - **State (S):** A representation of the current situation of the environment.
      - **Action (A):** A choice made by the agent.
      - **Reward (R):** Feedback from the environment indicating the immediate desirability of an action taken in a state.
      - **Policy (π):** The agent's strategy for choosing actions in given states.
      - **Value Function (V/Q):** Predicts the expected future reward from a state or state-action pair.
   - **Goal:** Learn an optimal policy `π*`.
   - **Examples:** Game playing (AlphaGo), robotics (learning to walk), resource management, recommendation systems.
   - **Common Algorithms:** Q-Learning, SARSA, Deep Q-Networks (DQN), Policy Gradients (e.g., REINFORCE, A2C, A3C), Proximal Policy Optimization (PPO).

---

### By Models

This classification focuses on the characteristics and assumptions of the model itself.

#### 1. Probabilistic vs. Non-probabilistic (Deterministic) Models

   - **Probabilistic Models:**
      - **Definition:** These models explicitly represent uncertainty by outputting probabilities or probability distributions over outcomes. They often make assumptions about the underlying probability distributions of the data (e.g., Gaussian, Bernoulli).
      - **Output:** A probability distribution `P(Y|X)` for classification, or a predictive distribution for regression (mean and variance).
      - **Advantages:** Can quantify uncertainty, often more interpretable, can be easily incorporated into Bayesian frameworks.
      - **Examples:**
         - Naive Bayes (Classification)
         - Logistic Regression (Outputs probabilities for classes)
         - Gaussian Mixture Models (Clustering, Density Estimation)
         - Hidden Markov Models (HMMs)
         - Bayesian Networks
         - Linear Regression (can be framed probabilistically by assuming Gaussian noise)
         - Gaussian Processes (Regression, Classification)

   - **Non-probabilistic (Deterministic) Models:**
      - **Definition:** These models provide a single, deterministic prediction without an explicit representation of uncertainty about the prediction itself (though some can provide confidence scores).
      - **Output:** A specific class label or a point estimate for regression.
      - **Advantages:** Can be simpler, sometimes computationally faster, may achieve high accuracy without explicit probabilistic assumptions.
      - **Examples:**
         - k-Nearest Neighbors (k-NN)
         - Support Vector Machines (SVMs - standard formulation, though Platt scaling can add probabilistic outputs)
         - Decision Trees (standard CART, ID3)
         - Perceptron
      - **Note:** The line can be blurry. For instance, SVM outputs can be calibrated to probabilities (Platt Scaling), and Decision Trees can be adapted to output class probabilities based on leaf node distributions.

#### 2. Linear vs. Non-linear Models

   - **Linear Models:**
      - **Definition:** Assume a linear relationship between the input features and the output. The decision boundary (for classification) or the regression fit is a hyperplane (a line in 2D, a plane in 3D, etc.).
      - **Mathematical Form (Regression):** `y = β₀ + β₁x₁ + ... + βₚxₚ + ε`
      - **Mathematical Form (Classification):** Decision based on `sign(β₀ + β₁x₁ + ... + βₚxₚ)`
      - **Advantages:** Simple, interpretable, computationally efficient, less prone to overfitting with small datasets.
      - **Examples:**
         - Linear Regression
         - Logistic Regression
         - Perceptron
         - Linear SVMs
         - Linear Discriminant Analysis (LDA)

   - **Non-linear Models:**
      - **Definition:** Can capture complex, non-linear relationships between input features and the output. The decision boundary or regression fit can be a curve or a more complex shape.
      - **Advantages:** More flexible, can model intricate patterns, often achieve higher accuracy on complex datasets.
      - **Disadvantages:** Can be more prone to overfitting, less interpretable, computationally more intensive.
      - **Examples:**
         - Polynomial Regression
         - Kernel SVMs (e.g., RBF kernel)
         - Decision Trees and ensemble methods (Random Forests, Gradient Boosting)
         - k-Nearest Neighbors (k-NN)
         - Neural Networks (especially Deep Learning)
         - Gaussian Processes

#### 3. Parametric vs. Non-parametric Models

   - **Parametric Models:**
      - **Definition:** Assume a specific functional form for the mapping function and have a fixed number of parameters, regardless of the size of the training data. The learning process involves estimating these parameters from the data.
      - **Characteristics:** Make strong assumptions about the data's distribution or the relationship between variables. Once parameters are learned, the original data is not needed for prediction.
      - **Advantages:** Simpler, faster to train, require less data (if assumptions hold).
      - **Disadvantages:** Limited complexity, may underfit if assumptions are incorrect (model bias).
      - **Examples:**
         - Linear Regression (parameters: `β₀, β₁, ...`)
         - Logistic Regression (parameters: `β₀, β₁, ...`)
         - Naive Bayes (parameters: class priors, feature conditional probabilities)
         - Perceptron
         - Neural Networks (fixed architecture defines a set of parameters - weights and biases)

   - **Non-parametric Models:**
      - **Definition:** Do not make strong assumptions about the functional form of the mapping function. The number of "parameters" or the model complexity can grow with the size of the training data.
      - **Characteristics:** More flexible, can fit a wider range of data distributions. Often, they keep (a subset of) the training data to make predictions.
      - **Advantages:** Can model complex relationships, less prone to underfitting due to restrictive assumptions.
      - **Disadvantages:** Require more data, computationally more expensive, more prone to overfitting if not regularized.
      - **Examples:**
         - k-Nearest Neighbors (k-NN) (parameters: the entire training dataset, `k`)
         - Decision Trees (complexity grows with data)
         - Support Vector Machines (non-linear kernels; support vectors effectively act as parameters determined by data)
         - Kernel Regression
         - Gaussian Processes (parameters define covariance function, but effective complexity grows)
         - Random Forests

   - **Note:** "Non-parametric" doesn't mean "no parameters" but rather that the parameters are not fixed in advance and can adapt in number or nature to the data.

---

### By Algorithms - Training Scheme

This classification refers to how the algorithm processes data for training.

#### 1. Online Learning
   - **Definition:** The model is trained incrementally by processing data instances (or small mini-batches) sequentially. It updates its parameters after each instance or mini-batch.
   - **Characteristics:**
      - Suitable for streaming data where data arrives continuously.
      - Can adapt to changes in the data distribution over time (concept drift).
      - Memory efficient, as it doesn't need the entire dataset in memory.
      - Can be used for very large datasets that don't fit in memory.
   - **Examples:** Stochastic Gradient Descent (SGD) is inherently online, Perceptron algorithm, online versions of SVMs, incremental decision trees.

#### 2. Batch Learning (Offline Learning)
   - **Definition:** The model is trained on the entire available training dataset at once.
   - **Characteristics:**
      - Simpler to implement and analyze for many algorithms.
      - Assumes the entire dataset is available before training begins.
      - If new data arrives, the model typically needs to be retrained from scratch on the full (old + new) dataset, which can be computationally expensive.
   - **Examples:** Standard Gradient Descent (using the full batch), training a Decision Tree on all data, traditional SVM training.
   - **Mini-batch Learning:** A common hybrid approach, especially in deep learning. The data is divided into small batches, and the model parameters are updated after processing each mini-batch. It combines some advantages of both online (efficiency, handling large data) and batch (more stable gradient estimates than pure online) learning.

---

### By Techniques - Underlying Methodologies

This focuses on core mathematical or philosophical approaches employed.

#### 1. Bayesian Learning
   - **Definition:** A probabilistic approach to learning that uses Bayes' Theorem to update the probability for a hypothesis (e.g., model parameters `θ`) as more evidence (data `D`) becomes available.
   - **Core Idea:** Start with a prior belief about parameters `P(θ)`, combine it with the likelihood of observing the data given parameters `P(D|θ)`, to obtain a posterior belief about parameters `P(θ|D)`.
   - **Bayes' Theorem:** `P(θ|D) = [P(D|θ) * P(θ)] / P(D)`
      - `P(θ|D)`: Posterior probability of parameters given data.
      - `P(D|θ)`: Likelihood of data given parameters.
      - `P(θ)`: Prior probability of parameters.
      - `P(D)`: Marginal likelihood of data (evidence).
   - **Advantages:** Principled way to incorporate prior knowledge, naturally handles uncertainty, provides full posterior distributions for parameters and predictions.
   - **Challenges:** Can be computationally intensive (especially calculating `P(D)` or sampling from complex posteriors), choice of prior can be influential.
   - **Examples:**
      - Naive Bayes Classifier
      - Bayesian Linear Regression
      - Bayesian Networks
      - Gaussian Processes
      - Latent Dirichlet Allocation (LDA) for topic modeling
      - Variational Autoencoders (VAEs) often have a Bayesian interpretation.

#### 2. Kernel Methods
   - **Definition:** A class of algorithms for pattern analysis whose best-known member is the Support Vector Machine (SVM). The general task is to find and study general types of relations (e.g., clusters, rankings, principal components, correlations, classifications) in datasets.
   - **Core Idea (Kernel Trick):** Perform a non-linear mapping of the input data into a high-dimensional feature space where linear methods can be applied. The kernel function `K(xᵢ, xⱼ) = φ(xᵢ) · φ(xⱼ)` computes the dot product of data points in this high-dimensional space `φ(x)` *without* explicitly computing the coordinates `φ(x)`. This avoids the computational cost of explicit transformation.
   - **Characteristics:**
      - Allows linear algorithms to learn non-linear functions or decision boundaries.
      - The choice of kernel function (e.g., Linear, Polynomial, Radial Basis Function (RBF), Sigmoid) is crucial.
   - **Advantages:** Effective in high-dimensional spaces, can model complex non-linear relationships, robust to overfitting with proper regularization and kernel choice.
   - **Examples:**
      - Support Vector Machines (SVMs) with kernels
      - Kernel PCA (Principal Component Analysis)
      - Kernel Ridge Regression
      - Gaussian Processes (kernels define the covariance function)
      - Multiple Kernel Learning

---

## Representation, Inference, and Learning (Troika)

This conceptual framework, popularized by Pedro Domingos, describes the essential ingredients of any machine learning algorithm:

### 1. Representation
   - **Definition:** How the knowledge or model is represented. It's the choice of the hypothesis space, i.e., the set of possible models or functions that the learner can learn.
   - **Question:** What form can the learned model take?
   - **Examples:**
      - **Linear models:** `y = wx + b` (parameters `w`, `b`)
      - **Decision trees:** A tree structure with if-then rules.
      - **Neural networks:** A network of interconnected nodes with weights and activation functions.
      - **Support Vector Machines:** A hyperplane defined by support vectors.
      - **Probabilistic models (e.g., Naive Bayes):** Conditional probability tables or distributions.
      - **Instance-based learners (e.g., k-NN):** The training instances themselves.
   - **Impact:** The choice of representation determines the types of patterns the model can learn and its complexity.

### 2. Inference (Evaluation / Prediction)
   - **Definition:** How to use the learned model to make predictions on new, unseen data, or to evaluate its "goodness." For probabilistic models, this often involves inferring probabilities or distributions.
   - **Question:** Given a learned model, how do we obtain outputs for new inputs? Or, how do we score a candidate model?
   - **Examples:**
      - **Linear models:** Compute `wx + b`.
      - **Decision trees:** Traverse the tree based on feature values.
      - **Neural networks:** Forward propagation through the network.
      - **Probabilistic models:** Use Bayes' rule or other probabilistic inference methods to compute `P(Y|X)`.
      - **Objective/Loss Function:** A function that measures how well the model's predictions match the true values (e.g., Mean Squared Error, Cross-Entropy Loss). This is often used during *learning* to guide optimization, but the act of applying it to *evaluate* a candidate model is part of this component.

### 3. Learning (Optimization / Search)
   - **Definition:** The process of finding the best model (parameters) within the chosen representation, typically by optimizing an objective function (e.g., minimizing a loss function or maximizing a likelihood function) based on the training data.
   - **Question:** How do we find the "best" model from the hypothesis space given the data?
   - **Examples:**
      - **Gradient Descent:** Iteratively adjust parameters to minimize a loss function.
      - **Closed-form solutions:** e.g., Ordinary Least Squares for Linear Regression.
      - **Greedy search:** e.g., In decision tree construction (ID3, C4.5, CART).
      - **Combinatorial optimization:** e.g., For some discrete problems.
      - **Expectation-Maximization (EM):** For models with latent variables.
      - **Markov Chain Monte Carlo (MCMC):** For Bayesian inference.
   - **Impact:** The choice of optimization algorithm determines the efficiency and effectiveness of the learning process and whether a good (or optimal) model can be found.

---

## Conclusion

Understanding these different categorizations provides a robust framework for navigating the vast landscape of machine learning. It's important to note that these categories are not always mutually exclusive; a single ML algorithm can often be described using terms from multiple categories. For example, Logistic Regression is a supervised, probabilistic, linear, parametric model, often trained using batch or mini-batch gradient descent. The choice of which type of ML approach to use depends heavily on the nature of the data, the problem at hand, computational resources, and the desired interpretability of the model.