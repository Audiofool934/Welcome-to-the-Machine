https://zhuanlan.zhihu.com/p/29846048

#### h. Singular Value Decomposition (SVD)

Singular Value Decomposition is a fundamental matrix factorization technique with wide-ranging applications in linear algebra, data science, and engineering. It asserts that any $m \times n$ matrix $\mathbf{A}$, regardless of whether it's square or rectangular, real or complex, can be decomposed into the product of three specific matrices.

**The Core Factorization:**

Any $m \times n$ matrix $\mathbf{A}$ can be factorized as:
$$ \mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T $$
(For complex matrices, this becomes $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^H$, where $H$ denotes the conjugate transpose.)

Let's break down each component:

1.  **$\mathbf{U}$ (Left Singular Vectors):**
    *   An $m \times m$ orthogonal matrix (if $\mathbf{A}$ is real) or unitary matrix (if $\mathbf{A}$ is complex).
    *   Orthogonal means $\mathbf{U}^T\mathbf{U} = \mathbf{U}\mathbf{U}^T = \mathbf{I}_m$ (where $\mathbf{I}_m$ is the $m \times m$ identity matrix).
    *   The columns of $\mathbf{U}$ are called the **left singular vectors** of $\mathbf{A}$. Let these be $\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_m$.
    *   These vectors form an orthonormal basis for the column space of $\mathbf{A}$ (and its orthogonal complement if $m > \text{rank}(\mathbf{A})$).
    *   Specifically, $\mathbf{u}_i$ are the eigenvectors of $\mathbf{A}\mathbf{A}^T$.

2.  **$\mathbf{\Sigma}$ (Singular Values):**
    *   An $m \times n$ rectangular diagonal matrix. This means only the entries $\Sigma_{ii}$ can be non-zero.
    *   The diagonal entries $\sigma_i = \Sigma_{ii}$ are called the **singular values** of $\mathbf{A}$.
    *   They are non-negative and conventionally sorted in descending order:
        $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$, where $r = \text{rank}(\mathbf{A})$.
    *   If $r < \min(m, n)$, the remaining singular values $\sigma_{r+1}, \dots, \sigma_{\min(m,n)}$ are zero.
    *   The structure of $\mathbf{\Sigma}$ depends on the relationship between $m$ and $n$:
        *   If $m=n$ (square A): $\mathbf{\Sigma} = \text{diag}(\sigma_1, \dots, \sigma_n)$.
        *   If $m > n$ (tall A): $\mathbf{\Sigma} = \begin{pmatrix} \text{diag}(\sigma_1, \dots, \sigma_n) \\ \mathbf{0}_{(m-n) \times n} \end{pmatrix}$.
        *   If $m < n$ (wide A): $\mathbf{\Sigma} = \begin{pmatrix} \text{diag}(\sigma_1, \dots, \sigma_m) & \mathbf{0}_{m \times (n-m)} \end{pmatrix}$.
    *   The number of non-zero singular values is equal to the rank of the matrix $\mathbf{A}$.

3.  **$\mathbf{V}^T$ (Transpose of Right Singular Vectors):**
    *   An $n \times n$ orthogonal matrix (if $\mathbf{A}$ is real) or unitary matrix (if $\mathbf{A}$ is complex).
    *   Orthogonal means $\mathbf{V}^T\mathbf{V} = \mathbf{V}\mathbf{V}^T = \mathbf{I}_n$.
    *   The columns of $\mathbf{V}$ (which are the rows of $\mathbf{V}^T$) are called the **right singular vectors** of $\mathbf{A}$. Let these be $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n$.
    *   These vectors form an orthonormal basis for the row space of $\mathbf{A}$ (and its orthogonal complement, the null space of $\mathbf{A}$, if $n > \text{rank}(\mathbf{A})$).
    *   Specifically, $\mathbf{v}_i$ are the eigenvectors of $\mathbf{A}^T\mathbf{A}$.

**Relationship to Eigenvalues:**

The singular values $\sigma_i$ of $\mathbf{A}$ are directly related to the eigenvalues of $\mathbf{A}^T\mathbf{A}$ and $\mathbf{A}\mathbf{A}^T$. Both $\mathbf{A}^T\mathbf{A}$ (an $n \times n$ matrix) and $\mathbf{A}\mathbf{A}^T$ (an $m \times m$ matrix) are symmetric (or Hermitian for complex $\mathbf{A}$) and positive semi-definite, meaning their eigenvalues are real and non-negative.

*   **For $\mathbf{A}^T\mathbf{A}$:**
    $\mathbf{A}^T\mathbf{A} = (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T)^T (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T) = (\mathbf{V}\mathbf{\Sigma}^T\mathbf{U}^T) (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T)$
    Since $\mathbf{U}^T\mathbf{U} = \mathbf{I}_m$:
    $\mathbf{A}^T\mathbf{A} = \mathbf{V}\mathbf{\Sigma}^T\mathbf{\Sigma}\mathbf{V}^T$
    This is an eigendecomposition of $\mathbf{A}^T\mathbf{A}$. The columns of $\mathbf{V}$ are the eigenvectors of $\mathbf{A}^T\mathbf{A}$, and the diagonal entries of $\mathbf{\Sigma}^T\mathbf{\Sigma}$ (which are $\sigma_i^2$) are the corresponding eigenvalues.
    So, $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^T\mathbf{A})}$, where $\lambda_i(\mathbf{A}^T\mathbf{A})$ are the eigenvalues of $\mathbf{A}^T\mathbf{A}$.

*   **For $\mathbf{A}\mathbf{A}^T$:**
    $\mathbf{A}\mathbf{A}^T = (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T) (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T)^T = (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T) (\mathbf{V}\mathbf{\Sigma}^T\mathbf{U}^T)$
    Since $\mathbf{V}^T\mathbf{V} = \mathbf{I}_n$:
    $\mathbf{A}\mathbf{A}^T = \mathbf{U}\mathbf{\Sigma}\mathbf{\Sigma}^T\mathbf{U}^T$
    This is an eigendecomposition of $\mathbf{A}\mathbf{A}^T$. The columns of $\mathbf{U}$ are the eigenvectors of $\mathbf{A}\mathbf{A}^T$, and the diagonal entries of $\mathbf{\Sigma}\mathbf{\Sigma}^T$ (which are $\sigma_i^2$) are the corresponding eigenvalues.
    So, $\sigma_i = \sqrt{\lambda_i(\mathbf{A}\mathbf{A}^T)}$, where $\lambda_i(\mathbf{A}\mathbf{A}^T)$ are the eigenvalues of $\mathbf{A}\mathbf{A}^T$.

The non-zero eigenvalues of $\mathbf{A}^T\mathbf{A}$ and $\mathbf{A}\mathbf{A}^T$ are the same and are equal to $\sigma_i^2$.

**Thin SVD (or Economy SVD):**

If $m > n$ (more rows than columns), $\mathbf{U}$ is $m \times m$ and $\mathbf{\Sigma}$ is $m \times n$. Many columns of $\mathbf{U}$ will multiply zero blocks in $\mathbf{\Sigma}$. We can define a "thin" SVD:
$\mathbf{A} = \mathbf{U}_{m \times n} \mathbf{\Sigma}_{n \times n} \mathbf{V}^T_{n \times n}$
Where:
*   $\mathbf{U}_{m \times n}$ (often denoted $\hat{\mathbf{U}}$) consists of the first $n$ columns of $\mathbf{U}$.
*   $\mathbf{\Sigma}_{n \times n}$ (often denoted $\hat{\mathbf{\Sigma}}$) is the top $n \times n$ block of $\mathbf{\Sigma}$.
*   $\mathbf{V}^T$ remains $n \times n$.

Similarly, if $m < n$ (more columns than rows):
$\mathbf{A} = \mathbf{U}_{m \times m} \mathbf{\Sigma}_{m \times m} \mathbf{V}^T_{m \times n}$
Where:
*   $\mathbf{U}$ remains $m \times m$.
*   $\mathbf{\Sigma}_{m \times m}$ is the left $m \times m$ block of $\mathbf{\Sigma}$.
*   $\mathbf{V}^T_{m \times n}$ (whose corresponding $\mathbf{V}_{n \times m}$ is often denoted $\hat{\mathbf{V}}$) consists of the first $m$ rows of $\mathbf{V}^T$ (or first $m$ columns of $\mathbf{V}$).

In general, if $r = \text{rank}(\mathbf{A})$, we can write a "compact SVD":
$\mathbf{A} = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$
where $\mathbf{U}_r$ is $m \times r$, $\mathbf{\Sigma}_r$ is $r \times r$ (diagonal with $\sigma_1, \dots, \sigma_r$), and $\mathbf{V}_r^T$ is $r \times n$.

**Geometric Interpretation:**
The SVD provides a geometric understanding of a linear transformation $\mathbf{x} \mapsto \mathbf{Ax}$. It decomposes the transformation into three simpler operations:
1.  **Rotation/Reflection ($\mathbf{V}^T$):** An orthogonal transformation that aligns the input basis vectors with the principal axes (right singular vectors).
2.  **Scaling ($\mathbf{\Sigma}$):** Scales the transformed basis vectors along these principal axes by the singular values. Some dimensions might be stretched, some shrunk, and some collapsed (if $\sigma_i=0$).
3.  **Rotation/Reflection ($\mathbf{U}$):** Another orthogonal transformation that aligns the scaled axes with the output basis vectors (left singular vectors).

---

**Applications:**

SVD is incredibly versatile. Here are some key applications:

1.  **Dimensionality Reduction (PCA derived from SVD):**
    Principal Component Analysis (PCA) aims to find a lower-dimensional subspace that captures the most variance in the data.
    *   Let $\mathbf{X}$ be an $m \times n$ data matrix where $m$ is the number of samples and $n$ is the number of features. Assume $\mathbf{X}$ is mean-centered (subtract the mean of each feature from all its values).
    *   The covariance matrix is $\mathbf{C} = \frac{1}{m-1}\mathbf{X}^T\mathbf{X}$. PCA involves finding the eigenvectors of $\mathbf{C}$.
    *   Using SVD of $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$:
        $\mathbf{X}^T\mathbf{X} = (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T)^T (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T) = \mathbf{V}\mathbf{\Sigma}^T\mathbf{U}^T\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T = \mathbf{V}(\mathbf{\Sigma}^T\mathbf{\Sigma})\mathbf{V}^T$.
    *   So, $\mathbf{C} = \frac{1}{m-1}\mathbf{V}(\mathbf{\Sigma}^T\mathbf{\Sigma})\mathbf{V}^T$.
    *   The columns of $\mathbf{V}$ are the eigenvectors of $\mathbf{X}^T\mathbf{X}$ (and thus of $\mathbf{C}$). These are the **principal components** (directions of maximum variance).
    *   The eigenvalues of $\mathbf{C}$ are $\lambda_i = \frac{\sigma_i^2}{m-1}$, where $\sigma_i$ are singular values of $\mathbf{X}$.
    *   To reduce dimensionality to $k < n$, we select the first $k$ columns of $\mathbf{V}$ (denoted $\mathbf{V}_k$) corresponding to the $k$ largest singular values.
    *   The transformed (projected) data in the lower-dimensional space is $\mathbf{Z} = \mathbf{X}\mathbf{V}_k$.
    *   Using SVD, $\mathbf{Z} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T\mathbf{V}_k = \mathbf{U}_k\mathbf{\Sigma}_k$, where $\mathbf{U}_k$ are the first $k$ columns of $\mathbf{U}$ and $\mathbf{\Sigma}_k$ is the top-left $k \times k$ block of $\mathbf{\Sigma}$ (or more precisely, the first $k$ columns of $\mathbf{U\Sigma}$). The columns of $\mathbf{U\Sigma}$ are the "principal component scores."

2.  **Matrix Approximation (Low-Rank Approximation):**
    The Eckart-Young-Mirsky theorem states that the best rank-$k$ approximation of a matrix $\mathbf{A}$ (in terms of Frobenius norm or spectral norm) is obtained by using its SVD.
    *   Given $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$.
    *   The best rank-$k$ approximation $\mathbf{A}_k$ is formed by taking the sum of the first $k$ terms, corresponding to the $k$ largest singular values:
        $$ \mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = \mathbf{U}_k \mathbf{\Sigma}_k \mathbf{V}_k^T $$
        where $\mathbf{U}_k$ are the first $k$ columns of $\mathbf{U}$, $\mathbf{\Sigma}_k$ is the $k \times k$ diagonal matrix of the first $k$ singular values, and $\mathbf{V}_k^T$ are the first $k$ rows of $\mathbf{V}^T$.
    *   This $\mathbf{A}_k$ minimizes $||\mathbf{A} - \mathbf{A}_k||_F$ (Frobenius norm) and $||\mathbf{A} - \mathbf{A}_k||_2$ (spectral norm) among all rank-$k$ matrices.
    *   **Example (Image Compression):** An image can be represented as a matrix of pixel values. By computing its SVD and keeping only the top $k$ singular values and corresponding singular vectors, we get a compressed version of the image. Higher $k$ means better quality but less compression.

3.  **Pseudo-Inverse Computation:**
    For any $m \times n$ matrix $\mathbf{A}$, its Moore-Penrose pseudo-inverse $\mathbf{A}^+$ can be computed using SVD.
    *   If $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$, then $\mathbf{A}^+ = \mathbf{V}\mathbf{\Sigma}^+\mathbf{U}^T$.
    *   $\mathbf{\Sigma}^+$ is an $n \times m$ matrix formed by taking the transpose of $\mathbf{\Sigma}$ and then taking the reciprocal of its non-zero diagonal entries.
        If $\Sigma_{ii} = \sigma_i \ne 0$, then $\Sigma^+_{ii} = 1/\sigma_i$.
        If $\Sigma_{ii} = 0$, then $\Sigma^+_{ii} = 0$.
    *   The pseudo-inverse is crucial for solving linear systems that are underdetermined or overdetermined.

4.  **Solving Least Squares Problems:**
    Consider the problem of finding $\mathbf{x}$ that minimizes $||\mathbf{Ax} - \mathbf{b}||_2$.
    *   The solution is given by $\mathbf{x} = \mathbf{A}^+\mathbf{b}$.
    *   Using SVD: $\mathbf{x} = (\mathbf{V}\mathbf{\Sigma}^+\mathbf{U}^T)\mathbf{b}$.
    *   This approach is numerically more stable than solving the normal equations $(\mathbf{A}^T\mathbf{A})\mathbf{x} = \mathbf{A}^T\mathbf{b}$, especially when $\mathbf{A}$ is ill-conditioned (has a high condition number, i.e., $\sigma_1/\sigma_r$ is large). SVD allows one to inspect singular values and potentially regularize by setting small singular values (or their reciprocals) to zero in $\mathbf{\Sigma}^+$, which can stabilize the solution.

5.  **Low-Rank Adaptation (LoRA) for Fine-tuning Large Models:**
    LoRA is a technique used to efficiently fine-tune large pre-trained models (like language models or diffusion models) by adapting only a small number of parameters.
    *   **Context:** Pre-trained models have weight matrices $\mathbf{W}_0$ (e.g., in a linear layer, $\mathbf{y} = \mathbf{W}_0\mathbf{x}$). Full fine-tuning updates all parameters in $\mathbf{W}_0$, which can be computationally expensive and require storing a full copy of the updated weights $\mathbf{W} = \mathbf{W}_0 + \Delta\mathbf{W}$.
    *   **LoRA Hypothesis:** The change in weights $\Delta\mathbf{W}$ during adaptation has a low "intrinsic rank." That is, $\Delta\mathbf{W}$ can be well-approximated by a low-rank matrix.
    *   **Mechanism:** LoRA freezes the original weights $\mathbf{W}_0$ and injects trainable, rank-decomposition matrices. Specifically, it models $\Delta\mathbf{W}$ as the product of two smaller matrices:
        $$ \Delta\mathbf{W} = \mathbf{B}\mathbf{A} $$
        (Note: some literature might use $\Delta\mathbf{W} = \mathbf{A}\mathbf{B}$. The order determines dimensions. Let's assume $\mathbf{W}_0$ is $d_{out} \times d_{in}$. Then $\mathbf{A}$ is $r \times d_{in}$ and $\mathbf{B}$ is $d_{out} \times r$, where $r$ is the rank of the adaptation, $r \ll \min(d_{out}, d_{in})$.)
    *   The modified forward pass becomes: $\mathbf{y} = \mathbf{W}_0\mathbf{x} + \mathbf{B}\mathbf{A}\mathbf{x}$.
    *   **Training:** Only $\mathbf{A}$ and $\mathbf{B}$ are trained. $\mathbf{A}$ is typically initialized with random Gaussian values, and $\mathbf{B}$ is initialized with zeros, so $\Delta\mathbf{W}$ is zero at the beginning of training.
    *   **Benefits:**
        *   **Parameter Efficiency:** The number of trainable parameters is $r \times d_{in} + d_{out} \times r$, which is much smaller than $d_{out} \times d_{in}$ if $r$ is small.
        *   **Storage Efficiency:** Only $\mathbf{A}$ and $\mathbf{B}$ need to be stored for each fine-tuned task, not the entire $\Delta\mathbf{W}$.
        *   **Task Switching:** Multiple tasks can be adapted from the same base model by swapping out their respective $\mathbf{A}$ and $\mathbf{B}$ matrices.
        *   **No Inference Latency:** Once trained, $\mathbf{W}' = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$ can be computed and used directly, incurring no extra latency compared to the original model.
    *   **Connection to SVD:** While LoRA doesn't directly compute SVD during training, the underlying principle is that the update $\Delta\mathbf{W}$ can be effectively represented by a low-rank structure. SVD tells us that any matrix *can* be decomposed, and its optimal low-rank approximation is given by truncating singular values. LoRA *learns* a task-specific low-rank update that is effective for fine-tuning.

In summary, SVD is a powerful tool that reveals the underlying structure of a matrix through its singular values and singular vectors. Its ability to decompose a matrix into orthogonal transformations and scaling makes it invaluable for understanding linear transformations, approximating data, solving systems of equations, and even efficiently adapting large-scale machine learning models.