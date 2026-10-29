# APEX-QoS: Server Response Time Prediction Using Multiple Linear Regression

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Statsmodels](https://img.shields.io/badge/statsmodels-0.14%2B-informational)](https://www.statsmodels.org/)
[![HTML5 / CSS3 / Vanilla JS](https://img.shields.io/badge/Frontend-Vanilla%20JS%20%2F%20Canvas-38bdf8)](index.html)
[![Academic Project](https://img.shields.io/badge/Academic-Micro%20Project%2013-success)](#academic-context)

An end-to-end mathematical data science and cloud telemetry project that predicts server response times using **Multiple Linear Regression (MLR)** and **Ordinary Least Squares (OLS)** inference. 

The system couples a rigorous statistical Python modeling engine with an interactive real-time web dashboard (**APEX-QoS**) and an automated 10-slide PowerPoint generator with LaTeX-rendered mathematical formulas.

---

## Table of Contents

- [Project Motivation & Overview](#project-motivation--overview)
- [Mathematical Framework](#mathematical-framework)
  - [1. Queueing Theory Formulation](#1-queueing-theory-formulation)
  - [2. Multiple Linear Regression Model](#2-multiple-linear-regression-model)
  - [3. OLS Estimation (Normal Equations)](#3-ols-estimation-normal-equations)
  - [4. Statistical Hypothesis Testing & Validation](#4-statistical-hypothesis-testing--validation)
- [Key Empirical Findings: Scenario A vs Scenario B](#key-empirical-findings-scenario-a-vs-scenario-b)
- [System Architecture](#system-architecture)
  - [1. Statistical Python Modeling Pipeline (`server_response_time_regression.py`)](#1-statistical-python-modeling-pipeline)
  - [2. Interactive Web Analytics Dashboard (`index.html`)](#2-interactive-web-analytics-dashboard)
  - [3. Automated PowerPoint Slide Generator (`create_presentation.py`)](#3-automated-powerpoint-slide-generator)
- [Diagnostic Visualizations (`outputs/`)](#diagnostic-visualizations-outputs)
- [Project Directory Structure](#project-directory-structure)
- [Installation & Quickstart](#installation--quickstart)
- [Course Outcomes & Academic Alignment](#course-outcomes--academic-alignment)
- [License](#license)

---

## Project Motivation & Overview

In high-throughput web architectures, cloud services, and microservices environments, unexpected latency surges cause severe degradation in Quality of Service (QoS) and costly Service Level Agreement (SLA) breaches. Traditional monitoring frameworks are **reactive**—they record high latency only after clients have experienced delays.

**APEX-QoS** replaces reactive monitoring with **proactive statistical inference**:
- Ingests pre-execution workload telemetry (CPU load, memory allocation, active concurrent sessions, network bandwidth, system throughput, error frequency, and task waiting queue times).
- Fits Ordinary Least Squares regression to predict end-to-end latency before job completion.
- Triggers dynamic architectural remediation (autoscaling pods, activating edge caches, or enforcing traffic rate-limiting) when latency exceeds the critical threshold ($3000\text{ ms}$).

---

## Mathematical Framework

### 1. Queueing Theory Formulation
In cloud task scheduling (Kaggle Cloud Workload Dataset, $N = 5,000$), total latency is defined by queueing theory:
$$\text{Response Time } (Y) = \text{Task Waiting Time } (W) + \text{Task Execution Time } (S)$$
Where:
- $W$: Queue delay spent waiting for an available CPU worker thread (ms).
- $S$: Service duration spent executing the task instructions on the compute cluster (ms).

### 2. Multiple Linear Regression Model
Given $k$ explanatory workload features $\mathbf{x} = [X_1, X_2, \dots, X_k]^T$:
$$Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \dots + \beta_k X_{ik} + \epsilon_i, \quad i = 1, \dots, n$$

In matrix notation:
$$\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$$

### 3. OLS Estimation (Normal Equations)
The coefficient vector $\hat{\boldsymbol{\beta}}$ minimizes the Sum of Squared Residuals ($\text{SSR}$):
$$\text{SSR}(\boldsymbol{\beta}) = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$$

Setting the gradient to zero yields the **Normal Equations**:
$$\mathbf{X}^T \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T \mathbf{y} \implies \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### 4. Statistical Hypothesis Testing & Validation
- **Individual Parameter Significance ($t$-test):**
  $$H_0: \beta_j = 0 \quad \text{vs} \quad H_1: \beta_j \ne 0, \quad t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim t_{n - k - 1}$$
- **Overall Model Significance ($F$-test / ANOVA):**
  $$H_0: \beta_1 = \dots = \beta_k = 0, \quad F = \frac{\text{MSR}}{\text{MSE}} = \frac{\text{SSR} / k}{\text{SSE} / (n - k - 1)} \sim F_{k, \, n - k - 1}$$
- **Goodness of Fit:**
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}, \quad R^2_{\text{adj}} = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - k - 1}\right]$$
- **Residual Homoscedasticity:**
  $$\text{Var}(e_i \mid \mathbf{X}) = \sigma^2 \quad \forall i, \quad e_i = y_i - \hat{y}_i$$

---

## Key Empirical Findings: Scenario A vs Scenario B

The central scientific discovery of this project is documented in the empirical comparison between **Scenario A** and **Scenario B**:

| Metric | Scenario A (Workload Parameters Only) | Scenario B (Workload + Predicted Execution Duration) |
| :--- | :---: | :---: |
| **Features Included** | CPU, Memory, Users, Bandwidth, Throughput, Error Rate, Queue Time | All Workload Features + Predicted Execution Duration ($\hat{S}$) |
| **Coefficient of Determination ($R^2$)** | **$0.002$** | **$0.944$ – $0.978$** |
| **Adjusted $R^2$** | $0.001$ | $0.977$ |
| **Root Mean Squared Error (RMSE)** | $1,410.2\text{ ms}$ | **$174.6\text{ ms}$** |
| **Mean Absolute Error (MAE)** | $1,192.5\text{ ms}$ | **$122.3\text{ ms}$** |
| **ANOVA $F$-Statistic** | $1.82$ ($p = 0.089$, not significant) | **$31,200.4$ ($p < 10^{-120}$, highly significant)** |
| **Regression Fit Plot** | Flat horizontal line at sample mean ($\approx 2978\text{ ms}$) | Tightly clustered along ideal diagonal ($y = x$) |

### Engineering Takeaway
External workload parameters (CPU, active users, queue delays) establish system pressure, but **task computational complexity** (execution duration) accounts for the largest proportion of variance in end-to-end response times. Combining queue metrics with predicted execution duration is essential for mission-critical QoS guarantees.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           APEX-QoS PROJECT SYSTEM                              │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌──────────────────┐          ┌──────────────────┐          ┌──────────────────┐
│  Python Backend  │          │   Web Platform   │          │  PowerPoint Deck │
│  (Data Science)  │          │  (Real-Time QoS) │          │  (Presentation)  │
├──────────────────┤          ├──────────────────┤          ├──────────────────┤
│ server_response_ │          │ index.html       │          │ create_          │
│ time_regression  │          │ styles.css       │          │ presentation.py  │
│ .py              │          │ app.js           │          │                  │
├──────────────────┤          ├──────────────────┤          ├──────────────────┤
│ • OLS Estimation │          │ • Live Sliders   │          │ • 10 Slide Deck  │
│ • ANOVA & F-test │          │ • 60 FPS Canvases│          │ • LaTeX Formulas │
│ • Cross-Valid.   │          │ • 2D Heatmap     │          │ • Embedded Plots │
│ • Outputs Export │          │ • 4K Lightbox    │          │ • Clean Layout   │
└────────┬─────────┘          └────────┬─────────┘          └────────┬─────────┘
         │                             │                             │
         ▼                             ▼                             ▼
   outputs/*.png              Client Browser Screen           *.pptx Presentation
 (8 High-Res Figures)          (Local Execution)               (Academic Defense)
```

### 1. Statistical Python Modeling Pipeline
[`server_response_time_regression.py`](server_response_time_regression.py)
- Ingests and cleans `cloud.csv` ($N = 5,000$) and `labeled_dataset.csv`.
- Executes IQR outlier screening, D'Agostino normality testing, and Pearson correlation matrices.
- Trains Simple Linear Regression, Multiple Linear Regression, and Random Forest benchmark models.
- Performs 5-fold cross-validation and extracts OLS summary statistics ($t$-scores, $p$-values, $95\%$ confidence intervals).
- Automatically writes 8 high-resolution diagnostic figures into `outputs/` and launches an interactive 2×2 diagnostic window on screen via `plt.show()`.

### 2. Interactive Web Analytics Dashboard
[`index.html`](index.html), [`styles.css`](styles.css), [`app.js`](app.js)
- **High-DPI Retina Canvases:**
  1. *Actual vs Predicted Latency:* Dual-scenario comparison rendering Scenario A (blue flat band) and Scenario B (green diagonal cluster) with live workload point crosshairs.
  2. *Residual Plot ($e = y - \hat{y}$):* Homoscedastic purple scatter centered on zero with dynamic drop-line.
  3. *Workload Correlation Heatmap Matrix:* Full 11×11 2D Pearson matrix with color gradient, toggleable to live horizontal impact bars.
  4. *Response Time Distribution:* 35-bin histogram with continuous Gaussian density fit and real-time $Z$-score / percentile sweep line.
- **Outputs Gallery & Lightbox:** Directly embeds all 8 generated Python diagnostic plots with click-to-enlarge 4K modal inspection and PNG download.
- **SLA Alert Engine:** Evaluates response time in real time and classifies QoS state (Safe $<1500\text{ ms}$, Elevated $1500\text{--}3000\text{ ms}$, SLA Breach $>3000\text{ ms}$).

### 3. Automated PowerPoint Slide Generator
[`create_presentation.py`](create_presentation.py)
- Builds a 10-slide academic defense presentation (`Micro_Project_13_Presentation.pptx`).
- Pre-renders mathematical equations into high-resolution transparent PNG images using Computer Modern typography (`mathtext.fontset = 'cm'`).
- Follows classroom presentation conventions (Title, Problem Statement, Objectives, Methodology, Mathematical Modeling, Data Analysis, Regression Results, Conclusion, References).

---

## Diagnostic Visualizations (`outputs/`)

All analytical plots generated by the Python modeling pipeline are stored in [`outputs/`](outputs):

| Filename | Diagnostic Focus | Description |
| :--- | :--- | :--- |
| [`00_master_regression_dashboard.png`](outputs/00_master_regression_dashboard.png) | **Master 2×2 Dashboard** | Unified diagnostic grid: Distribution, 2D Correlation Heatmap, Actual vs Predicted (A vs B), and Residuals. |
| [`01_response_time_distribution.png`](outputs/01_response_time_distribution.png) | **Distribution & Normality** | 40-bin response time histogram with continuous Gaussian normal density overlay. |
| [`02_correlation_heatmap.png`](outputs/02_correlation_heatmap.png) | **Bivariate Multicollinearity** | Full Pearson correlation matrix between all 10 compute parameters and response time. |
| [`03_scatter_workload.png`](outputs/03_scatter_workload.png) | **Trellis Bivariate Scatters** | Linearity evaluation across CPU Utilization, Memory Consumption, and Active Users. |
| [`A_actual_vs_predicted.png`](outputs/A_actual_vs_predicted.png) | **Scenario A Fit ($R^2 = 0.002$)** | Demonstrates the horizontal failure line when task duration is omitted. |
| [`A_residuals.png`](outputs/A_residuals.png) | **Scenario A Residuals** | Shows heavy heteroscedastic spread without duration features. |
| [`B_actual_vs_predicted.png`](outputs/B_actual_vs_predicted.png) | **Scenario B Fit ($R^2 = 0.944$)** | Tight scatter alignment along the 45-degree diagonal line $y = x$. |
| [`B_residuals.png`](outputs/B_residuals.png) | **Scenario B Residuals** | Homoscedastic zero-mean distribution validating OLS error assumptions. |

---

## Project Directory Structure

```text
micro-pro/
├── index.html                           # APEX-QoS interactive analytics dashboard
├── styles.css                           # Glassmorphic dark design system & lightbox styles
├── app.js                               # Real-time regression engine & canvas visualizer
│
├── server_response_time_regression.py   # Core Python statistical modeling pipeline
├── create_presentation.py               # Automated PowerPoint slide builder (python-pptx)
├── generate_ppt.py                      # One-line invocation wrapper for presentation build
│
├── cloud.csv                            # Kaggle Cloud Workload Dataset (N=5,000 records)
├── labeled_dataset.csv                  # Secondary Web Performance Dataset
├── requirements.txt                     # Python package dependencies
├── LICENSE                              # MIT Open Source License
├── README.md                            # Comprehensive project documentation
│
├── outputs/                             # Generated high-resolution diagnostic plots
│   ├── 00_master_regression_dashboard.png
│   ├── 01_response_time_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_scatter_workload.png
│   ├── A_actual_vs_predicted.png
│   ├── A_residuals.png
│   ├── B_actual_vs_predicted.png
│   └── B_residuals.png
│
├── PROJECT_REPORT.md                    # Formal academic project documentation
├── VIVA_VOCE_GUIDE.md                   # Comprehensive viva examination Q&A guide
└── PRESENTATION_SLIDES.md               # Presentation slide script & reference content
```

---

## Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- Modern web browser (Chrome, Edge, Firefox, or Safari)

### 1. Clone & Set Up Virtual Environment
```powershell
# Navigate into the project folder
cd "path/to/micro pro"

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run the Statistical Regression Engine
```powershell
python server_response_time_regression.py
```
> **Behavior:** Computes OLS summaries, cross-validates models, saves all figures to `outputs/`, and pops up the interactive 2×2 diagnostic window on screen. Closing the window finishes the terminal output.

### 4. Launch the Interactive Web Dashboard
```powershell
Start-Process index.html
```
> **Behavior:** Opens the APEX-QoS dashboard locally in your default web browser. Adjust the telemetry sliders to view real-time latency forecasts, canvas updates, and inspect the `outputs/` gallery.

### 5. Generate PowerPoint Presentation Slides
```powershell
python create_presentation.py
```
> **Behavior:** Synthesizes the 10-slide academic slide deck with rendered LaTeX mathematical equations and exports `Micro_Project_13_Presentation.pptx`.

---

## Course Outcomes & Academic Alignment

This project satisfies the following Course Outcomes (COs) for Data Science and Computer Science curricula:

- **CO2 (Descriptive Statistics & Correlation):** Measures central tendency (Mean $= 2978.8\text{ ms}$), variance, D'Agostino skewness, and computes bivariate Pearson correlation matrices across compute telemetry.
- **CO3 (Regression Modeling & OLS Estimation):** Formulates Normal Equations to fit Multiple Linear Regression models and estimates coefficients with Ordinary Least Squares.
- **CO4 (Hypothesis Testing & ANOVA):** Conducts individual parameter $t$-tests ($p < 0.001$) and overall model $F$-tests ($F = 31200.4, p < 10^{-120}$) with $95\%$ confidence intervals.
- **CO5 (Residual Diagnostics & Validation):** Diagnoses homoscedasticity, residual error normality, and performs 5-fold cross-validation ($R^2_{\text{CV}} = 0.976$).

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

```text
MIT License
Copyright (c) 2026 Dharanidharan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions...
```
#   S e r v e r - R e s p o n s e - T i m e - P r e d i c t i o n  
 