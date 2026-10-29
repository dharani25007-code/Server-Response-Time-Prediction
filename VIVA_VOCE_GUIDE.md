# Viva Voce Questions & Answers Guide
## Micro Project 13: Server Response Time Prediction Using Regression Models

This guide prepares you for questions commonly asked by professors and examiners during project evaluations and viva voce examinations.

---

### Q1: What is the core objective of this project?
**Answer:**  
The objective is to predict server response time in web applications using regression models based on real-time workload telemetry (such as CPU utilization, memory consumption, concurrent active users, bandwidth, throughput, and queue delay). This allows web operators to proactively forecast SLA violations and trigger autoscaling before latency degrades.

---

### Q2: Why is Response Time defined as Task Waiting Time + Task Execution Time?
**Answer:**  
In queueing theory ($M/M/1$ or $G/G/c$ queues), a request experiences two phases:
1. **Waiting Time ($W_q$):** The delay spent waiting in the server or socket queue before a worker thread is assigned.
2. **Execution Time ($T_s$):** The actual compute time required to execute application logic, perform database queries, and stream the response.  
Total latency observed by a web client is the sum: $T_{\text{response}} = T_{\text{wait}} + T_{\text{execution}}$.

---

### Q3: How do you map this project to Course Outcomes CO2 through CO5?
**Answer:**  
- **CO2 (Two-Dimensional Random Variables & Correlation):** Bivariate correlation analysis (Pearson $r$) between workload variables and response time, joint scatter distributions, and testing normality of response times.
- **CO3 (Statistical Inference & Hypothesis Testing):** Testing the statistical significance of regression coefficients using Student's $t$-tests ($H_0: \beta_j = 0$), ANOVA $F$-test for overall model validity ($p < 10^{-94}$), and computing 95% Confidence Intervals.
- **CO4 (Regression Analysis & Goodness of Fit):** Deriving Ordinary Least Squares (OLS) normal equations $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$, comparing Simple Linear Regression ($R^2 = 0.32$) against Multiple Linear Regression ($R^2 = 0.68$), and computing RMSE and MAE.
- **CO5 (Applied Predictive Modeling & Quality of Service):** Verifying Gauss-Markov residual assumptions (homoscedasticity, zero mean), benchmarking against Random Forest, and embedding the model into a live web QoS monitoring system with SLA threshold alerts.

---

### Q4: How is the OLS regression formula $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ derived?
**Answer:**  
We formulate the Sum of Squared Residuals:  
$$S(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{y}^T \mathbf{y} - 2\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X}\boldsymbol{\beta}$$  
Differentiating with respect to $\boldsymbol{\beta}$ and setting the gradient to zero:  
$$\frac{\partial S}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T \mathbf{y} + 2\mathbf{X}^T \mathbf{X}\boldsymbol{\beta} = 0 \implies \mathbf{X}^T \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T \mathbf{y}$$  
Multiplying by $(\mathbf{X}^T \mathbf{X})^{-1}$ gives the closed-form solution: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$.

---

### Q5: What is the difference between $R^2$ and Adjusted $R^2$?
**Answer:**  
- **$R^2$ (Coefficient of Determination):** Measures the proportion of variance in response time explained by the features ($R^2 = 1 - \frac{\text{SSE}}{\text{SST}}$). However, $R^2$ never decreases when adding more variables, even irrelevant noise.
- **Adjusted $R^2$:** Penalizes the model for adding non-informative predictors by accounting for degrees of freedom:  
  $$R^2_{\text{adj}} = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - p - 1}\right]$$  
In our model, $R^2 = 0.684$ and $R^2_{\text{adj}} = 0.681$, showing minimal penalty and indicating that our features are genuinely informative.

---

### Q6: What does the $F$-test in regression tell you?
**Answer:**  
The $F$-test tests the null hypothesis that **all** regression coefficients are simultaneously zero:  
$$H_0: \beta_1 = \beta_2 = \dots = \beta_p = 0 \quad \text{vs.} \quad H_1: \text{At least one } \beta_j \neq 0$$  
Our model yielded $F = 1,542.8$ with a $p$-value $< 10^{-94}$. Because $p < 0.05$, we reject $H_0$, proving that the regression relationship is statistically significant and not an artifact of random sampling.

---

### Q7: What are the key Gauss-Markov assumptions, and how did you verify them?
**Answer:**  
Under the Gauss-Markov theorem, OLS estimators are the Best Linear Unbiased Estimator (BLUE) if:
1. **Linearity:** Relationship between $X$ and $Y$ is linear in parameters.
2. **Zero Conditional Mean:** $\mathbb{E}[\varepsilon_i | X] = 0$. (Verified: mean residual $= 0.000$ ms).
3. **Homoscedasticity:** Constant error variance $\text{Var}(\varepsilon_i) = \sigma^2$. (Verified: residual plot displays uniform rectangular dispersion without fan/cone shapes).
4. **No Multicollinearity:** Predictors are not perfectly linearly dependent.
5. **Normality of Errors:** $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. (Verified: residual histogram shows bell curve with skewness $= 0.082$).

---

### Q8: Why does System Throughput have a negative coefficient ($\beta = -48.20$)?
**Answer:**  
Throughput is the rate at which the server completes tasks per second. When system throughput is high, tasks are being cleared efficiently from worker queues, leading to shorter wait times and lower overall latency. When throughput drops (e.g. under lock contention or I/O waits), queues back up, driving response time up.

---

### Q9: Why compare Scenario A against Scenario B?
**Answer:**  
- **Scenario A (Workload only):** Simulates what a load balancer or monitoring agent observes before a task finishes (CPU, memory, bandwidth, queue delay). It achieves $R^2 = 0.684$.
- **Scenario B (Workload + Predicted Execution Duration):** Adds an upstream job profiler's estimate of task complexity. This drives $R^2$ up to $0.978$ and reduces RMSE to $174.6$ ms.  
Demonstrating both scenarios shows both baseline telemetry prediction and enriched profiling architectures.

---

### Q10: How does this model integrate into intelligent cloud monitoring?
**Answer:**  
In production, a sidecar monitoring agent streams server telemetry into the lightweight regression formula. If predicted latency exceeds the SLA target (e.g., $3000$ ms), the system automatically:
1. Triggers the **Horizontal Pod Autoscaler (HPA)** to spawn additional containers before requests queue up.
2. Informs edge load balancers to route traffic away from saturated nodes.
3. Activates aggressive caching and rate limiting.
