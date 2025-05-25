### 4. Why Cross-Entropy (CE) as the Cost Function?

For Linear Regression, we used Mean Squared Error (MSE). For Logistic Regression, directly using MSE on the sigmoid output leads to a non-convex cost function. Instead, Logistic Regression (and many other classification models outputting probabilities) uses **Cross-Entropy Loss** (also known as Log Loss).

---

#### **Side Note: Understanding Entropy, Cross-Entropy, and KL Divergence in the Context of Loss Functions**

To fully appreciate why Cross-Entropy is a suitable loss function for classification models that output probabilities, it's helpful to understand some concepts from information theory: Entropy, Cross-Entropy, and Kullback-Leibler (KL) Divergence.

**a. Entropy (熵)**

*   **Motivation & Intuition:**
    Entropy, in information theory, measures the **average level of "surprise" or "uncertainty"** inherent in a random variable's possible outcomes.
    -   If a probability distribution is highly concentrated (e.g., one outcome is almost certain), the uncertainty is low, and thus the entropy is low. You're not very "surprised" when that outcome occurs.
    -   If a probability distribution is uniform (e.g., a fair coin flip where heads and tails are equally likely), the uncertainty is high, and the entropy is high. You are more "surprised" by the outcome.
    Entropy quantifies the average number of bits needed to encode or transmit messages drawn from this distribution if using an optimal coding scheme.

*   **Definition:**
    For a discrete random variable $X$ with possible outcomes $\{x_1, x_2, \dots, x_n\}$ and a probability mass function $P(X=x_i) = p_i$, the entropy $H(P)$ or $H(X)$ is defined as:
    $$ H(P) = H(X) = - \sum_{i=1}^{n} p_i \log_b (p_i) $$
    -   The sum is over all possible outcomes $i$.
    -   $p_i$ is the probability of the $i$-th outcome.
    -   $\log_b$ is the logarithm base.
        -   If $b=2$, entropy is measured in **bits**. This is common in information theory.
        -   If $b=e$ (natural logarithm, $\ln$), entropy is measured in **nats**. This is common in machine learning.
    -   By convention, $0 \log 0 = 0$, as $\lim_{p \to 0^+} p \log p = 0$. This handles cases where some outcomes have zero probability.

*   **Example:**
    -   A fair coin ($P(H)=0.5, P(T)=0.5$):
        $H(P) = - (0.5 \log_2 0.5 + 0.5 \log_2 0.5) = - (0.5 \cdot (-1) + 0.5 \cdot (-1)) = 1 \text{ bit}$.
        This is the maximum entropy for a two-outcome variable.
    -   A biased coin ($P(H)=0.9, P(T)=0.1$):
        $H(P) = - (0.9 \log_2 0.9 + 0.1 \log_2 0.1) \approx - (0.9 \cdot (-0.152) + 0.1 \cdot (-3.322)) \approx 0.1368 + 0.3322 \approx 0.469 \text{ bits}$.
        Lower entropy because the outcome is more predictable.

**b. Cross-Entropy (交叉熵)**

*   **Motivation & Intuition:**
    Cross-Entropy measures the **average number of bits needed to identify an event drawn from a set if a coding scheme is used that is optimized for an *estimated* probability distribution $Q$, rather than the *true* distribution $P$.**
    -   In machine learning for classification, $P$ represents the true underlying distribution of the labels (often a one-hot encoded vector, e.g., $[0, 1, 0]$ for class 1 out of 3).
    -   $Q$ represents the probability distribution predicted by our model (e.g., $[0.1, 0.7, 0.2]$ from a softmax output).
    -   Cross-Entropy tells us how "different" our model's predicted distribution $Q$ is from the true distribution $P$. If $Q$ is a good approximation of $P$, the cross-entropy will be low (close to the entropy of $P$). If $Q$ is a poor approximation, the cross-entropy will be high.
    -   Therefore, minimizing cross-entropy encourages our model's predictions $Q$ to get closer to the true distribution $P$.

*   **Definition:**
    For two discrete probability distributions $P = \{p_1, \dots, p_n\}$ (true distribution) and $Q = \{q_1, \dots, q_n\}$ (estimated/predicted distribution) over the same set of events, the cross-entropy $H(P, Q)$ is defined as:
    $$ H(P, Q) = - \sum_{i=1}^{n} p_i \log_b (q_i) $$
    -   Note the difference from entropy: we use $p_i$ (from the true distribution) as the weight but take the logarithm of $q_i$ (from the predicted distribution).

*   **Connection to Logistic/Softmax Regression Loss:**
    Consider binary classification where the true label $y^{(i)}$ is either 0 or 1.
    -   If $y^{(i)}=1$, the true distribution $P$ is $(P(y=1)=1, P(y=0)=0)$.
    -   If $y^{(i)}=0$, the true distribution $P$ is $(P(y=1)=0, P(y=0)=1)$.
    Let our model predict $h_\theta(\mathbf{x}^{(i)}) = P(y=1|\mathbf{x}^{(i)})$ and $1-h_\theta(\mathbf{x}^{(i)}) = P(y=0|\mathbf{x}^{(i)})$. So $Q = (h_\theta(\mathbf{x}^{(i)}), 1-h_\theta(\mathbf{x}^{(i)}))$.

    If $y^{(i)}=1$: $P=(1,0)$. $Q=(h_\theta, 1-h_\theta)$.
    $H(P,Q) = - (1 \cdot \log(h_\theta(\mathbf{x}^{(i)})) + 0 \cdot \log(1-h_\theta(\mathbf{x}^{(i)}))) = -\log(h_\theta(\mathbf{x}^{(i)}))$.

    If $y^{(i)}=0$: $P=(0,1)$. $Q=(h_\theta, 1-h_\theta)$.
    $H(P,Q) = - (0 \cdot \log(h_\theta(\mathbf{x}^{(i)})) + 1 \cdot \log(1-h_\theta(\mathbf{x}^{(i)}))) = -\log(1-h_\theta(\mathbf{x}^{(i)}))$.

    These are exactly the terms inside the summation of the Cross-Entropy loss function we derived for Logistic Regression using MLE!
    $$ J(\mathbf{\theta}) = \frac{1}{m} \sum_{i=1}^{m} \text{Cost}(h_\theta(\mathbf{x}^{(i)}), y^{(i)}) = \frac{1}{m} \sum_{i=1}^{m} H(P^{(i)}, Q^{(i)}) $$
    Where $P^{(i)}$ is the true one-hot distribution for example $i$, and $Q^{(i)}$ is the predicted distribution.
    Similarly for multi-class classification with softmax, the loss term for a single sample (where true class is $c$) is $-\log(q_c)$, which is the cross-entropy between the one-hot true distribution and the softmax predicted distribution.

**c. Kullback-Leibler (KL) Divergence (KL散度 / 相对熵 - Xiāngduì Shāng)**

*   **Motivation & Intuition:**
    KL Divergence measures the **"distance" or "divergence" of one probability distribution $Q$ from a reference probability distribution $P$.** It quantifies how much information is lost when $Q$ is used to approximate $P$.
    -   $D_{KL}(P || Q) \ge 0$.
    -   $D_{KL}(P || Q) = 0$ if and only if $P=Q$ (the distributions are identical).
    -   Importantly, KL Divergence is **not symmetric**: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$ in general. So it's not a true distance metric but rather a measure of "directed divergence."

*   **Definition:**
    For discrete distributions $P$ and $Q$:
    $$ D_{KL}(P || Q) = \sum_{i=1}^{n} p_i \log_b \left(\frac{p_i}{q_i}\right) $$
    This can also be written as:
    $$ D_{KL}(P || Q) = \sum_{i=1}^{n} p_i (\log_b p_i - \log_b q_i) $$
    $$ D_{KL}(P || Q) = \left( - \sum_{i=1}^{n} p_i \log_b q_i \right) - \left( - \sum_{i=1}^{n} p_i \log_b p_i \right) $$
    $$ D_{KL}(P || Q) = H(P, Q) - H(P) $$

*   **Relationship to Cross-Entropy and Entropy:**
    From the last equation:
    **Cross-Entropy $H(P, Q)$ = Entropy $H(P)$ + KL Divergence $D_{KL}(P || Q)$**

*   **Why Minimize Cross-Entropy?**
    In machine learning supervised classification:
    -   $P$ is the true (empirical) distribution of the labels in our training data. This is fixed and given by the data itself. Thus, the entropy $H(P)$ of the true labels is a constant.
    -   $Q$ is the distribution predicted by our model, which depends on the model parameters $\mathbf{\theta}$.
    -   When we minimize the Cross-Entropy $H(P, Q)$ with respect to $\mathbf{\theta}$, since $H(P)$ is constant, we are effectively minimizing $D_{KL}(P || Q)$.
    -   Minimizing $D_{KL}(P || Q)$ means we are trying to make our model's predicted distribution $Q$ as "close" as possible to the true distribution $P$. This is precisely our goal in training a probabilistic classifier.

    Therefore, minimizing the Cross-Entropy loss is equivalent to minimizing the KL divergence between the predicted distribution and the true data distribution, which is a well-motivated objective from an information-theoretic perspective. It encourages the model to learn the true underlying probabilities of the classes.

---

The Cross-Entropy loss for Logistic Regression can be derived from the principle of Maximum Likelihood Estimation (MLE), as shown previously. The connection to KL Divergence and Entropy provides an alternative and often more general justification for its use. It highlights that we are trying to make our model's probability estimates match the true underlying probabilities as closely as possible.

This cost function $J(\mathbf{\theta})$ (Cross-Entropy) is **convex** for Logistic Regression (and Softmax Regression), ensuring that gradient descent can converge to the global minimum.

(... Rest of the section on gradient and multi-class CE ...)