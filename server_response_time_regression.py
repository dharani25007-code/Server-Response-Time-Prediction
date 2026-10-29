"""
Server Response Time Prediction Using Regression Models
=======================================================
Micro Project - Probability and Statistics for Data Science and Cyber Security

Datasets (put the CSV files in the same folder as this script, or pass paths):
  1. cloud.csv            -> Kaggle: Comprehensive Cloud Workload Performance
  2. labeled_dataset.csv  -> Kaggle: Website Performance dataset (optional)

Run:
    pip install pandas numpy scikit-learn matplotlib seaborn statsmodels scipy
    python server_response_time_regression.py
    python server_response_time_regression.py --cloud path/cloud.csv --web path/labeled_dataset.csv

All plots are saved in the ./outputs folder and all numbers are printed to the console.

IMPORTANT NOTE ABOUT THE CLOUD DATASET
--------------------------------------
cloud.csv has NO column literally called "response time". So we define it the
standard queueing way:
        Response Time = Task Waiting Time + Task Execution Time   (ms)
"""

import argparse
import os
import warnings

import matplotlib
# Enable interactive GUI display backend on Windows
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
OUT = "outputs"
RANDOM_STATE = 42

# Column names in cloud.csv
WAIT = "Task_Waiting_Time (ms)"
EXEC = "Task_Execution_Time (ms)"
PRED_EXEC = "Predicted_Execution_Duration (ms)"
TARGET = "Response_Time_ms"

# Pure workload / resource features (known BEFORE the task finishes)
WORKLOAD_FEATURES = [
    "CPU_Utilization (%)",
    "Memory_Consumption (MB)",
    "Number_of_Active_Users",
    "Network_Bandwidth_Utilization (Mbps)",
    "System_Throughput (tasks/sec)",
    "Error_Rate (%)",
    "Memory_Utilization (%)",
    "CPU_Share (%)",
    "Memory_Share (%)",
    "Bandwidth_Share (%)",
]


def header(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


# ----------------------------------------------------------------------------
# STEP 1-2: LOAD + PREPROCESS
# ----------------------------------------------------------------------------
def load_and_preprocess_cloud(path):
    header("STEP 1-2: DATA COLLECTION AND PREPROCESSING (cloud.csv)")
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    # Define response time = waiting + execution
    df[TARGET] = df[WAIT] + df[EXEC]

    print(f"Missing values : {int(df.isna().sum().sum())}")
    dups = df.duplicated().sum()
    print(f"Duplicate rows : {dups}")
    df = df.drop_duplicates()

    # Outlier detection using the IQR rule (reported, not blindly removed)
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[TARGET] < low) | (df[TARGET] > high)).sum())
    print(f"IQR outliers in response time: {n_out} (bounds {low:.0f} to {high:.0f} ms)")
    df = df[(df[TARGET] >= low) & (df[TARGET] <= high)]
    print(f"Rows after cleaning: {len(df)}")
    return df


# ----------------------------------------------------------------------------
# STEP 3: EXPLORATORY STATISTICAL ANALYSIS
# ----------------------------------------------------------------------------
def explore(df):
    header("STEP 3: EXPLORATORY STATISTICAL ANALYSIS")
    cols = [TARGET] + WORKLOAD_FEATURES + [PRED_EXEC, WAIT, EXEC]
    desc = df[cols].describe().T[["mean", "50%", "std", "min", "max"]]
    desc["variance"] = df[cols].var()
    print(desc.round(2).to_string())

    skew = stats.skew(df[TARGET])
    print(f"\nResponse time skewness: {skew:.3f}")
    stat, p = stats.normaltest(df[TARGET])
    print(f"Normality test (D'Agostino) p-value: {p:.4g} "
          f"-> {'approximately normal' if p > 0.05 else 'NOT normal'}")

    # Pearson correlation of every feature with response time
    corr = df[cols].corr()[TARGET].drop(TARGET).sort_values(ascending=False)
    print("\nPearson correlation with Response Time:")
    print(corr.round(3).to_string())

    # Plot 1: histogram
    plt.figure(figsize=(7, 4))
    sns.histplot(df[TARGET], bins=40, kde=True, color="#2a6fdb")
    plt.title("Distribution of Response Time")
    plt.xlabel("Response time (ms)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/01_response_time_distribution.png", dpi=150)
    plt.close()

    # Plot 2: correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_correlation_heatmap.png", dpi=150)
    plt.close()

    # Plot 3: scatter of response time vs key workload features
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, c in zip(ax, ["CPU_Utilization (%)", "Memory_Consumption (MB)",
                         "Number_of_Active_Users"]):
        a.scatter(df[c], df[TARGET], s=5, alpha=0.4)
        a.set_xlabel(c)
        a.set_ylabel("Response time (ms)")
    plt.suptitle("Response Time vs Workload Parameters")
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_scatter_workload.png", dpi=150)
    plt.close()
    return corr


# ----------------------------------------------------------------------------
# STEP 4-9: MODEL, TRAIN, EVALUATE, STATISTICAL VALIDATION
# ----------------------------------------------------------------------------
def run_scenario(df, features, name, tag):
    header(f"MODELLING - {name}")
    print("Features used:")
    for f in features:
        print("   -", f)

    X, y = df[features], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)
    print(f"\nTrain size: {len(X_tr)} | Test size: {len(X_te)} (80:20 split)")

    results = {}

    # Model 1: Simple linear regression on the single most correlated feature
    best_single = X_tr.corrwith(y_tr).abs().idxmax()
    simple = make_pipeline(StandardScaler(), LinearRegression())
    simple.fit(X_tr[[best_single]], y_tr)
    results["Simple Linear Regression"] = metrics(
        y_te, simple.predict(X_te[[best_single]]))
    print(f"\nSimple LR uses the strongest single feature: {best_single}")

    # Model 2: Multiple linear regression (main model of the project)
    multi = make_pipeline(StandardScaler(), LinearRegression())
    multi.fit(X_tr, y_tr)
    pred_multi = multi.predict(X_te)
    results["Multiple Linear Regression"] = metrics(y_te, pred_multi)

    # Model 3: Random forest (non-linear comparison model)
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE,
                               n_jobs=-1)
    rf.fit(X_tr, y_tr)
    results["Random Forest (comparison)"] = metrics(y_te, rf.predict(X_te))

    # Comparison table
    table = pd.DataFrame(results).T
    cv = cross_val_score(make_pipeline(StandardScaler(), LinearRegression()),
                         X, y, cv=5, scoring="r2")
    print("\nTest-set performance:")
    print(table.round(3).to_string())
    print(f"\n5-fold cross-validated R2 (multiple LR): "
          f"{cv.mean():.3f} +/- {cv.std():.3f}")

    # Statistical validation with OLS: t-tests, p-values, 95% CI
    ols = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
    print("\nOLS regression summary (training data):")
    print(ols.summary().tables[1])
    print(f"F-statistic = {ols.fvalue:.2f}, p-value = {ols.f_pvalue:.4g}")
    print(f"R2 = {ols.rsquared:.4f}, Adjusted R2 = {ols.rsquared_adj:.4f}")
    sig = ols.pvalues.drop("const")
    sig = sig[sig < 0.05]
    print("Statistically significant features (p < 0.05): "
          + (", ".join(sig.index) if len(sig) else "NONE"))

    # Plots
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_te, pred_multi, s=8, alpha=0.5)
    lim = [y.min(), y.max()]
    plt.plot(lim, lim, "r--", label="Perfect prediction")
    plt.xlabel("Actual response time (ms)")
    plt.ylabel("Predicted response time (ms)")
    plt.title(f"Actual vs Predicted - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}_actual_vs_predicted.png", dpi=150)
    plt.close()

    resid = y_te - pred_multi
    plt.figure(figsize=(6, 4))
    plt.scatter(pred_multi, resid, s=8, alpha=0.5)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted response time (ms)")
    plt.ylabel("Residual (ms)")
    plt.title(f"Residual Plot - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}_residuals.png", dpi=150)
    plt.close()

    return multi, table, (X_te, y_te, pred_multi)


def predict_new(model, features, values, label):
    """Predict the response time for a new workload condition."""
    row = pd.DataFrame([values], columns=features)
    print(f"[{label}] predicted response time = {model.predict(row)[0]:.1f} ms")


# ----------------------------------------------------------------------------
# OPTIONAL: SECOND DATASET (real website measurements)
# ----------------------------------------------------------------------------
def website_dataset(path):
    header("SECOND DATASET: labeled_dataset.csv (real website measurements)")
    d = pd.read_csv(path).dropna()
    d = d[d["Response Time(s)"] > 0]  # removes invalid negative/zero readings
    print(f"Rows after cleaning: {len(d)}")

    feats = ["Page Size (KB)", "Load Time(s)", "Throughput"]
    target = "Response Time(s)"
    print("\nPearson correlation with response time:")
    print(d[feats + [target]].corr()[target].drop(target).round(3).to_string())

    # Response time is heavily right-skewed, so we also try a log transform
    X = d[feats]
    for label, y in [("raw response time", d[target]),
                     ("log(response time)", np.log(d[target]))]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE)
        m = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
        r2 = r2_score(y_te, m.predict(X_te))
        print(f"Multiple LR on {label}: test R2 = {r2:.3f}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default="cloud.csv")
    ap.add_argument("--web", default="labeled_dataset.csv")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    df = load_and_preprocess_cloud(args.cloud)
    explore(df)

    # Scenario A: pure workload parameters only (the project's original idea)
    model_a, table_a, test_a = run_scenario(
        df, WORKLOAD_FEATURES, "Scenario A: workload parameters only", "A")

    # Scenario B: add the dataset's own predicted execution duration
    feats_b = WORKLOAD_FEATURES + [PRED_EXEC]
    model_b, table_b, test_b = run_scenario(
        df, feats_b, "Scenario B: workload + predicted execution duration", "B")

    header("PREDICTING FOR A NEW WORKLOAD (example)")
    sample = df[feats_b].median().to_dict()
    sample["CPU_Utilization (%)"] = 90
    sample["Number_of_Active_Users"] = 4500
    predict_new(model_a, WORKLOAD_FEATURES,
                [sample[f] for f in WORKLOAD_FEATURES], "Scenario A")
    predict_new(model_b, feats_b, [sample[f] for f in feats_b], "Scenario B")

    if os.path.exists(args.web):
        website_dataset(args.web)
    else:
        print(f"\n(Skipping second dataset: {args.web} not found)")

    header("FINAL COMPARISON (test R2 of Multiple Linear Regression)")
    print(f"Scenario A (workload only)        : "
          f"{table_a.loc['Multiple Linear Regression', 'R2']:.3f}")
    print(f"Scenario B (+ predicted duration) : "
          f"{table_b.loc['Multiple Linear Regression', 'R2']:.3f}")

    # Display interactive master dashboard on screen
    display_and_save_dashboard(df, model_a, model_b, test_a, test_b)


def display_and_save_dashboard(df, model_a, model_b, test_a, test_b):
    """
    Display key diagnostic regression figures in an interactive window on screen,
    and save the master dashboard figure to outputs/.
    """
    X_te_a, y_te_a, pred_a = test_a
    X_te_b, y_te_b, pred_b = test_b

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    try:
        fig.canvas.manager.set_window_title("Micro Project 13: Regression Models Diagnostic Dashboard")
    except Exception:
        pass

    # 1. Distribution of Response Time
    ax1 = axes[0, 0]
    sns.histplot(df[TARGET], bins=35, kde=True, color="#2563eb", ax=ax1)
    ax1.set_title("Response Time Distribution & Normal Fit", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Response time (ms)")
    ax1.set_ylabel("Frequency")

    # 2. Pearson Correlation Heatmap
    ax2 = axes[0, 1]
    cols = [TARGET] + WORKLOAD_FEATURES
    sns.heatmap(df[cols].corr(), cmap="coolwarm", center=0, annot=False, ax=ax2)
    ax2.set_title("Workload Correlation Heatmap", fontsize=11, fontweight="bold")

    # 3. Actual vs Predicted Latency (Scenario A vs B)
    ax3 = axes[1, 0]
    ax3.scatter(y_te_a, pred_a, s=10, alpha=0.45, color="#2563eb", label="Scenario A (Workload only)")
    ax3.scatter(y_te_b, pred_b, s=10, alpha=0.45, color="#10b981", label="Scenario B (Workload + Duration)")
    lim = [df[TARGET].min(), df[TARGET].max()]
    ax3.plot(lim, lim, "r--", linewidth=1.5, label="Ideal (y = x)")
    ax3.set_title("Actual vs Predicted Latency (OLS)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Actual response time (ms)")
    ax3.set_ylabel("Predicted response time (ms)")
    ax3.legend(fontsize=9)

    # 4. Residual Plot (Scenario A)
    ax4 = axes[1, 1]
    resid_a = y_te_a - pred_a
    ax4.scatter(pred_a, resid_a, s=10, alpha=0.45, color="#8b5cf6")
    ax4.axhline(0, color="r", linestyle="--", linewidth=1.5)
    ax4.set_title("Residual Plot (Homoscedasticity Diagnosis)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Predicted response time (ms)")
    ax4.set_ylabel("Residual (e = y - ŷ)")

    plt.suptitle("Micro Project 13: Server Response Time Prediction — Diagnostic Dashboard",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()

    # 1. Save the master dashboard figure to outputs FIRST so files exist
    master_path = f"{OUT}/00_master_regression_dashboard.png"
    plt.savefig(master_path, dpi=200)

    print(f"\n{'='*75}")
    print(f"  [i] DISPLAYING INTERACTIVE DIAGNOSTIC PLOTS ON SCREEN...")
    print(f"      (Inspect the window; close it when finished to complete execution)")
    print(f"{'='*75}")

    # 2. POP UP AND DISPLAY ON SCREEN (blocks until user inspects/closes window)
    try:
        plt.show()
    except Exception as e:
        print(f"  [Note] Interactive window notice: {e}")

    # 3. ONLY AFTER DISPLAYING / CLOSING THE WINDOW, PRINT THE SAVE CONFIRMATION
    print(f"\nPlots saved in the '{OUT}' folder.")
    print(f"Master diagnostic dashboard saved to: {master_path}")
    print(f"All 8 individual & master diagnostic plots ready:")
    print(f"  • {OUT}/01_response_time_distribution.png")
    print(f"  • {OUT}/02_correlation_heatmap.png")
    print(f"  • {OUT}/03_scatter_workload.png")
    print(f"  • {OUT}/A_actual_vs_predicted.png")
    print(f"  • {OUT}/A_residuals.png")
    print(f"  • {OUT}/B_actual_vs_predicted.png")
    print(f"  • {OUT}/B_residuals.png")
    print(f"  • {OUT}/00_master_regression_dashboard.png")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
