"""Generate Week 4 lecture-support notebooks (clustering, forecasting, reporting)."""
from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).resolve().parents[1]
WEEK4_DIR = BASE / "Week_04_Unsupervised_Learning_Time_Series_and_Reporting"
LECTURE_DIR = WEEK4_DIR / "Lecture_Support"
DATA_DIR = BASE / "data" / "week4"
FIGURES_DIR = WEEK4_DIR / "figures"

for d in (LECTURE_DIR, DATA_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

COMMON_HELPER = '''from pathlib import Path

def find_repo_root():
    cwd = Path.cwd().resolve()
    for candidate in [cwd] + list(cwd.parents):
        if (candidate / "requirements.txt").exists() and (candidate / "data").exists():
            return candidate
    return cwd

REPO_ROOT = find_repo_root()
DATA_DIR = REPO_ROOT / "data" / "week4"
FIGURES_DIR = REPO_ROOT / "Week_04_Unsupervised_Learning_Time_Series_and_Reporting" / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CUSTOMER_CSV = DATA_DIR / "week4_customer_segmentation.csv"
AIRPASSENGERS_CSV = DATA_DIR / "week4_air_passengers.csv"
'''


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def build_day1():
    cells = [
        md(
            "# Week 4 Day 1 – Unsupervised Learning: Clustering + PCA\n\n"
            "## Learning objectives\n"
            "- Distinguish supervised and unsupervised learning\n"
            "- Apply K-Means and DBSCAN clustering\n"
            "- Choose clustering parameters using elbow, silhouette, and inspection\n"
            "- Use PCA to visualize clusters in 2D\n"
            "- Combine cluster labels with supervised models"
        ),
        md(
            "## Section 1 — Introduction to Unsupervised Learning\n\n"
            "**Supervised learning** uses a target variable (labels). The model learns to predict that target.\n\n"
            "**Unsupervised learning** has no target variable. We look for hidden structure in the data.\n\n"
            "**Clustering** groups similar observations together. There are no predefined labels — we must "
            "**interpret** what each cluster means in business or domain language.\n\n"
            "> **Why interpretation matters:** A cluster number (0, 1, 2) is not useful by itself. "
            "We must describe each group (for example: high spenders, occasional visitors)."
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "from sklearn.datasets import make_blobs\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.cluster import KMeans, DBSCAN\n"
            "from sklearn.decomposition import PCA\n"
            "from sklearn.metrics import silhouette_score\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\n"
            "%matplotlib inline\n"
            "sns.set_theme(style='whitegrid')\n"
            "RANDOM_STATE = 42"
        ),
        md("### Shared path helper"),
        code(COMMON_HELPER),
        md(
            "## Section 2 — Generate or Load Example Dataset\n\n"
            "We start with `make_blobs` to see clear groups, then build a realistic **customer segmentation** table "
            "(income, spending score, visit frequency, age)."
        ),
        code(
            "def create_customer_dataset(n_rows=400, seed=42):\n"
            "    rng = np.random.default_rng(seed)\n"
            "    centers = np.array([\n"
            "        [55_000, 85, 12, 34],\n"
            "        [32_000, 35, 4, 52],\n"
            "        [78_000, 55, 8, 41],\n"
            "    ])\n"
            "    labels_true = rng.choice([0, 1, 2], size=n_rows, p=[0.35, 0.35, 0.30])\n"
            "    rows = []\n"
            "    for lab in labels_true:\n"
            "        c = centers[lab]\n"
            "        rows.append([\n"
            "            rng.normal(c[0], 8_000),\n"
            "            np.clip(rng.normal(c[1], 12), 1, 100),\n"
            "            np.clip(rng.normal(c[2], 2), 1, 20),\n"
            "            np.clip(rng.normal(c[3], 8), 18, 75),\n"
            "        ])\n"
            "    arr = np.array(rows)\n"
            "    df = pd.DataFrame(arr, columns=['annual_income', 'spending_score', 'visit_frequency', 'age'])\n"
            "    df = df.round({'annual_income': 0, 'spending_score': 1, 'visit_frequency': 1, 'age': 1})\n"
            "    return df\n\n"
            "if CUSTOMER_CSV.exists():\n"
            "    customers = pd.read_csv(CUSTOMER_CSV)\n"
            "else:\n"
            "    customers = create_customer_dataset()\n"
            "    customers.to_csv(CUSTOMER_CSV, index=False)\n"
            "    print('Saved dataset to:', CUSTOMER_CSV)\n\n"
            "display(customers.head())\n"
            "print('\\nShape:', customers.shape)\n"
            "display(customers.describe().round(2))\n"
            "print('\\nMissing values:')\n"
            "print(customers.isna().sum())"
        ),
        code(
            "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\n"
            "cols = ['annual_income', 'spending_score', 'visit_frequency', 'age']\n"
            "for ax, col in zip(axes.ravel(), cols):\n"
            "    ax.hist(customers[col], bins=20, color='steelblue', edgecolor='white')\n"
            "    ax.set_title(col.replace('_', ' ').title())\n"
            "plt.suptitle('Feature Distributions', y=1.02)\n"
            "plt.tight_layout()\n"
            "plt.savefig(FIGURES_DIR / 'day1_feature_distributions.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md(
            "## Section 3 — Preprocessing for Clustering\n\n"
            "Distance-based algorithms (K-Means, DBSCAN) compare feature scales. "
            "If income is in thousands and spending score is 1–100, income dominates distance. "
            "**StandardScaler** centers each feature to mean 0 and unit variance."
        ),
        code(
            "feature_cols = ['annual_income', 'spending_score', 'visit_frequency', 'age']\n"
            "X = customers[feature_cols].copy()\n\n"
            "scaler = StandardScaler()\n"
            "X_scaled = scaler.fit_transform(X)\n"
            "X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)\n"
            "display(X_scaled_df.describe().round(2))"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
            "axes[0].boxplot([customers[c] for c in feature_cols], labels=feature_cols)\n"
            "axes[0].set_title('Before scaling')\n"
            "axes[0].tick_params(axis='x', rotation=30)\n"
            "axes[1].boxplot([X_scaled_df[c] for c in feature_cols], labels=feature_cols)\n"
            "axes[1].set_title('After StandardScaler')\n"
            "axes[1].tick_params(axis='x', rotation=30)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "## Section 4 — K-Means Clustering\n\n"
            "K-Means partitions data into **K** groups by minimizing within-cluster distance (inertia / SSE).\n\n"
            "- Choose **K** (number of clusters)\n"
            "- Initialize centroids\n"
            "- Assign points to nearest centroid\n"
            "- Update centroids as cluster means\n"
            "- Repeat until stable"
        ),
        code(
            "k_values = range(2, 9)\n"
            "inertias = []\n"
            "sil_scores = []\n\n"
            "for k in k_values:\n"
            "    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)\n"
            "    labels = km.fit_predict(X_scaled)\n"
            "    inertias.append(km.inertia_)\n"
            "    sil_scores.append(silhouette_score(X_scaled, labels))\n\n"
            "param_df = pd.DataFrame({'K': list(k_values), 'inertia': inertias, 'silhouette': sil_scores})\n"
            "display(param_df.round(3))"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
            "axes[0].plot(param_df['K'], param_df['inertia'], marker='o')\n"
            "axes[0].set_title('Elbow method (inertia / SSE)')\n"
            "axes[0].set_xlabel('K'); axes[0].set_ylabel('Inertia')\n"
            "axes[1].plot(param_df['K'], param_df['silhouette'], marker='o', color='darkorange')\n"
            "axes[1].set_title('Silhouette score vs K')\n"
            "axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette')\n"
            "plt.tight_layout()\n"
            "plt.savefig(FIGURES_DIR / 'day1_kmeans_parameter_selection.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        code(
            "best_k = int(param_df.sort_values('silhouette', ascending=False).iloc[0]['K'])\n"
            "print('Selected K (highest silhouette on this grid):', best_k)\n\n"
            "kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)\n"
            "customers['kmeans_cluster'] = kmeans.fit_predict(X_scaled)\n"
            "centers_original = scaler.inverse_transform(kmeans.cluster_centers_)\n"
            "centers_df = pd.DataFrame(centers_original, columns=feature_cols)\n"
            "centers_df.index.name = 'cluster'\n"
            "display(centers_df.round(1))\n"
            "print('Inertia:', round(kmeans.inertia_, 2))\n"
            "print('Silhouette:', round(silhouette_score(X_scaled, customers['kmeans_cluster']), 3))"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 5))\n"
            "scatter = ax.scatter(customers['annual_income'], customers['spending_score'],\n"
            "                     c=customers['kmeans_cluster'], cmap='tab10', alpha=0.75)\n"
            "ax.set_xlabel('Annual income'); ax.set_ylabel('Spending score')\n"
            "ax.set_title(f'K-Means clusters (K={best_k})')\n"
            "plt.colorbar(scatter, label='cluster')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### How to interpret K-Means clusters\n\n"
            "Read cluster **centers** in original units:\n"
            "- **Cluster 0 (example):** higher spending score and visit frequency → engaged shoppers\n"
            "- **Cluster 1:** lower income and spending → budget-conscious segment\n"
            "- **Cluster 2:** high income, moderate spending → potential upsell segment\n\n"
            "Always validate interpretations with plots and domain knowledge."
        ),
        md(
            "## Section 5 — DBSCAN\n\n"
            "DBSCAN groups dense regions and marks sparse points as **noise** (label −1).\n\n"
            "- **eps:** neighborhood radius\n"
            "- **min_samples:** points needed to form a dense region\n"
            "- **Core points:** enough neighbors within eps\n"
            "- **Border points:** near a core point\n"
            "- **Noise:** neither core nor border\n\n"
            "DBSCAN can find non-spherical clusters and outliers. It does **not** need K in advance."
        ),
        code(
            "def run_dbscan(eps, min_samples=5):\n"
            "    model = DBSCAN(eps=eps, min_samples=min_samples)\n"
            "    labels = model.fit_predict(X_scaled)\n"
            "    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)\n"
            "    n_noise = int((labels == -1).sum())\n"
            "    sil = np.nan\n"
            "    if n_clusters > 1:\n"
            "        mask = labels != -1\n"
            "        if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:\n"
            "            sil = silhouette_score(X_scaled[mask], labels[mask])\n"
            "    return labels, n_clusters, n_noise, sil\n\n"
            "eps_values = [0.35, 0.45, 0.55, 0.65, 0.75]\n"
            "db_rows = []\n"
            "for eps in eps_values:\n"
            "    labels, n_cl, n_noise, sil = run_dbscan(eps)\n"
            "    db_rows.append({'eps': eps, 'clusters': n_cl, 'noise_points': n_noise, 'silhouette': sil})\n"
            "dbscan_summary = pd.DataFrame(db_rows)\n"
            "display(dbscan_summary.round(3))"
        ),
        code(
            "db_eps = 0.55\n"
            "db_labels, n_db_clusters, n_db_noise, db_sil = run_dbscan(db_eps)\n"
            "customers['dbscan_cluster'] = db_labels\n"
            "print(f'DBSCAN (eps={db_eps}): clusters={n_db_clusters}, noise={n_db_noise}, silhouette={db_sil:.3f}')"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "axes[0].scatter(customers['annual_income'], customers['spending_score'],\n"
            "                c=customers['kmeans_cluster'], cmap='tab10', alpha=0.75)\n"
            "axes[0].set_title('K-Means')\n"
            "axes[1].scatter(customers['annual_income'], customers['spending_score'],\n"
            "                c=customers['dbscan_cluster'], cmap='tab10', alpha=0.75)\n"
            "axes[1].set_title('DBSCAN')\n"
            "for ax in axes:\n"
            "    ax.set_xlabel('Annual income'); ax.set_ylabel('Spending score')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "**DBSCAN does not require K, but it requires good `eps` and `min_samples`.** "
            "Too small `eps` → many noise points; too large → one giant cluster."
        ),
        md(
            "## Section 6 — Choosing Clustering Parameters\n\n"
            "**K-Means:** elbow (inertia), silhouette, interpretability, domain knowledge.\n\n"
            "**DBSCAN:** eps sensitivity, min_samples, noise percentage, visual inspection."
        ),
        code(
            "def summarize_dbscan_eps(eps_list, min_samples=5):\n"
            "    rows = []\n"
            "    for eps in eps_list:\n"
            "        _, n_cl, n_noise, sil = run_dbscan(eps, min_samples=min_samples)\n"
            "        rows.append({'eps': eps, 'clusters': n_cl, 'noise_points': n_noise, 'silhouette': sil})\n"
            "    return pd.DataFrame(rows)\n\n"
            "display(summarize_dbscan_eps([0.4, 0.5, 0.55, 0.6, 0.7]).round(3))"
        ),
        md(
            "### Common mistakes / warnings\n"
            "- Clustering on unscaled features when scales differ\n"
            "- Choosing K only from the elbow when clusters overlap\n"
            "- Treating DBSCAN noise (−1) as a normal segment\n"
            "- Forgetting that cluster IDs are arbitrary (0 vs 2 can swap between runs)"
        ),
        md(
            "## Section 7 — PCA for Cluster Visualization\n\n"
            "**PCA (Principal Component Analysis)** finds directions of maximum variance.\n"
            "- **PC1** and **PC2** are linear combinations of original features\n"
            "- **Explained variance ratio** shows how much information each PC keeps\n\n"
            "> **PCA does not create the clusters.** It helps us visualize them in 2D when we have many features."
        ),
        code(
            "pca = PCA(n_components=2, random_state=RANDOM_STATE)\n"
            "X_pca = pca.fit_transform(X_scaled)\n"
            "customers['PC1'] = X_pca[:, 0]\n"
            "customers['PC2'] = X_pca[:, 1]\n"
            "print('Explained variance ratio:', np.round(pca.explained_variance_ratio_, 3))\n"
            "print('Total variance explained (2 PCs):', round(pca.explained_variance_ratio_.sum(), 3))"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "for ax, col, title in zip(axes, ['kmeans_cluster', 'dbscan_cluster'], ['K-Means in PCA space', 'DBSCAN in PCA space']):\n"
            "    sc = ax.scatter(customers['PC1'], customers['PC2'], c=customers[col], cmap='tab10', alpha=0.75)\n"
            "    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_title(title)\n"
            "    plt.colorbar(sc, ax=ax, label='cluster')\n"
            "plt.tight_layout()\n"
            "plt.savefig(FIGURES_DIR / 'day1_pca_cluster_plots.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md(
            "## Section 8 — Combining Unsupervised and Supervised Workflows\n\n"
            "Workflow:\n"
            "1. Create a synthetic target (customer value)\n"
            "2. Train regression **without** cluster label\n"
            "3. Add cluster label as a feature and retrain\n"
            "4. Compare MAE / RMSE / R²"
        ),
        code(
            "rng = np.random.default_rng(RANDOM_STATE)\n"
            "customers['customer_value'] = (\n"
            "    0.02 * customers['annual_income']\n"
            "    + 45 * customers['spending_score']\n"
            "    + 80 * customers['visit_frequency']\n"
            "    - 5 * customers['age']\n"
            "    + rng.normal(0, 200, len(customers))\n"
            ")\n\n"
            "y = customers['customer_value']\n"
            "X_base = customers[feature_cols]\n"
            "X_with_cluster = customers[feature_cols + ['kmeans_cluster']]\n\n"
            "Xb_train, Xb_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=RANDOM_STATE)\n"
            "Xc_train, Xc_test, _, _ = train_test_split(X_with_cluster, y, test_size=0.2, random_state=RANDOM_STATE)\n\n"
            "def reg_metrics(model, X_tr, X_te, y_tr, y_te, name):\n"
            "    model.fit(X_tr, y_tr)\n"
            "    pred = model.predict(X_te)\n"
            "    return {\n"
            "        'Model': name,\n"
            "        'MAE': mean_absolute_error(y_te, pred),\n"
            "        'RMSE': np.sqrt(mean_squared_error(y_te, pred)),\n"
            "        'R2': r2_score(y_te, pred),\n"
            "    }\n\n"
            "rows = [\n"
            "    reg_metrics(LinearRegression(), Xb_train, Xb_test, y_train, y_test, 'Linear (no cluster)'),\n"
            "    reg_metrics(LinearRegression(), Xc_train, Xc_test, y_train, y_test, 'Linear (+ cluster)'),\n"
            "    reg_metrics(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), Xb_train, Xb_test, y_train, y_test, 'RF (no cluster)'),\n"
            "    reg_metrics(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE), Xc_train, Xc_test, y_train, y_test, 'RF (+ cluster)'),\n"
            "]\n"
            "supervised_cmp = pd.DataFrame(rows)\n"
            "display(supervised_cmp.round(3))"
        ),
        code(
            "# Optional: one model per cluster vs global model\n"
            "global_rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)\n"
            "global_rf.fit(X_base, y)\n"
            "global_pred = global_rf.predict(X_base)\n"
            "global_rmse = np.sqrt(mean_squared_error(y, global_pred))\n\n"
            "per_cluster_preds = np.zeros(len(customers))\n"
            "for cl in sorted(customers['kmeans_cluster'].unique()):\n"
            "    mask = customers['kmeans_cluster'] == cl\n"
            "    local = RandomForestRegressor(n_estimators=80, random_state=RANDOM_STATE)\n"
            "    local.fit(X_base[mask], y[mask])\n"
            "    per_cluster_preds[mask] = local.predict(X_base[mask])\n\n"
            "local_rmse = np.sqrt(mean_squared_error(y, per_cluster_preds))\n"
            "print('Global RF RMSE (in-sample demo):', round(global_rmse, 2))\n"
            "print('Per-cluster RF RMSE (in-sample demo):', round(local_rmse, 2))"
        ),
        md(
            "## Section 9 — Student Exercise\n\n"
            "1. Choose **K** using elbow and silhouette.\n"
            "2. Run K-Means with your chosen K.\n"
            "3. Run DBSCAN with at least two `eps` values and compare noise/clusters.\n"
            "4. Visualize clusters with PCA (PC1 vs PC2).\n"
            "5. Interpret each cluster in simple language.\n"
            "6. Write 3–5 sentences: which method worked better for this dataset and why?\n\n"
            "### Reflection questions\n"
            "- When would DBSCAN be preferable to K-Means?\n"
            "- What business action could you take for one cluster?\n"
            "- Did adding cluster labels improve supervised performance? Why might that happen?"
        ),
        md(
            "## Mini summary\n"
            "- Unsupervised learning discovers structure without labels.\n"
            "- Scale features before distance-based clustering.\n"
            "- K-Means needs K; DBSCAN needs eps and min_samples.\n"
            "- PCA helps visualization but does not define clusters.\n"
            "- Cluster labels can become features in supervised models."
        ),
        md(
            "## Final task checklist\n"
            "- [ ] K chosen and justified\n"
            "- [ ] K-Means and DBSCAN completed\n"
            "- [ ] PCA plot saved\n"
            "- [ ] Cluster interpretation written\n"
            "- [ ] Supervised comparison attempted"
        ),
        code("print('Week 4 Day 1 Clustering + PCA completed successfully.')"),
    ]
    write_notebook(LECTURE_DIR / "01_Week4_Day1_Clustering_PCA.ipynb", cells)


def build_day2():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from week4_day2_cells import get_day2_cells

    cells = get_day2_cells(md, code, COMMON_HELPER)
    write_notebook(LECTURE_DIR / "02_Week4_Day2_Time_Series_Forecasting.ipynb", cells)


def build_day3():
    cells = [
        md(
            "# Week 4 Day 3 – Reporting, Reproducibility and Final Project Preparation\n\n"
            "## Learning objectives\n"
            "- Explain why reporting and reproducibility matter in ML projects\n"
            "- Structure a final project folder and report\n"
            "- Fix random seeds and document environments\n"
            "- Combine clustering with supervised learning responsibly\n"
            "- Understand deployment and model maintenance concepts"
        ),
        md(
            "## Section 1 — Why Reporting Matters\n\n"
            "A strong machine learning project is not only about accuracy. It should be:\n\n"
            "- **Understandable** for stakeholders\n"
            "- **Reproducible** by teammates and future you\n"
            "- **Interpretable** with clear assumptions\n"
            "- **Honest** about limitations\n"
            "- **Useful** for real decisions"
        ),
        md(
            "## Section 2 — Reproducible Project Structure\n\n"
            "```text\n"
            "final_project/\n"
            "│\n"
            "├── data/\n"
            "│   ├── raw/\n"
            "│   └── processed/\n"
            "│\n"
            "├── notebooks/\n"
            "│   └── final_project_workflow.ipynb\n"
            "│\n"
            "├── src/\n"
            "│   ├── preprocessing.py\n"
            "│   ├── clustering.py\n"
            "│   ├── forecasting.py\n"
            "│   └── evaluation.py\n"
            "│\n"
            "├── figures/\n"
            "│\n"
            "├── reports/\n"
            "│   └── final_report.md\n"
            "│\n"
            "├── requirements.txt\n"
            "├── environment.yml\n"
            "├── README.md\n"
            "└── random_seed.txt\n"
            "```\n\n"
            "| Folder / file | Purpose |\n"
            "|---|---|\n"
            "| `data/raw` | Original downloads, never edited |\n"
            "| `data/processed` | Cleaned tables used by models |\n"
            "| `notebooks/` | Exploratory and teaching workflows |\n"
            "| `src/` | Reusable functions (preprocessing, metrics) |\n"
            "| `figures/` | Saved plots for reports |\n"
            "| `reports/` | Written interpretation for graders/stakeholders |\n"
            "| `requirements.txt` | Pip package list |\n"
            "| `random_seed.txt` | Documented seed for replication |"
        ),
        md("## Section 3 — Reproducibility Checklist"),
        code(
            "import sys\n"
            "import platform\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib\n"
            "import sklearn\n"
            "import statsmodels\n\n"
            "checklist = [\n"
            "    'Code available and runnable',\n"
            "    'Data source described',\n"
            "    'Preprocessing documented',\n"
            "    'Random seed fixed',\n"
            "    'Package versions recorded',\n"
            "    'Train/test split explained',\n"
            "    'Metrics clearly reported',\n"
            "    'Figures reproducible',\n"
            "    'Limitations discussed',\n"
            "    'GitHub link included',\n"
            "]\n"
            "for i, item in enumerate(checklist, 1):\n"
            "    print(f'{i:02d}. [ ] {item}')"
        ),
        code(COMMON_HELPER),
        code(
            "def pkg_version(mod, name=None):\n"
            "    name = name or mod.__name__\n"
            "    return getattr(mod, '__version__', 'unknown')\n\n"
            "versions = {\n"
            "    'Python': sys.version.split()[0],\n"
            "    'Platform': platform.platform(),\n"
            "    'numpy': pkg_version(np),\n"
            "    'pandas': pkg_version(pd),\n"
            "    'matplotlib': pkg_version(matplotlib),\n"
            "    'scikit-learn': pkg_version(sklearn),\n"
            "    'statsmodels': pkg_version(statsmodels),\n"
            "}\n"
            "try:\n"
            "    import xgboost\n"
            "    versions['xgboost'] = pkg_version(xgboost)\n"
            "except ImportError:\n"
            "    versions['xgboost'] = 'not installed'\n"
            "try:\n"
            "    import torch\n"
            "    versions['torch'] = pkg_version(torch)\n"
            "except ImportError:\n"
            "    versions['torch'] = 'not installed'\n\n"
            "env_df = pd.DataFrame(list(versions.items()), columns=['Package', 'Version'])\n"
            "display(env_df)\n"
            "env_df.to_csv(DATA_DIR / 'week4_environment_versions.csv', index=False)"
        ),
        md(
            "## Section 4 — Random Seeds and Reproducibility\n\n"
            "Many algorithms use randomness (initialization, sampling, bagging). "
            "Without a fixed **seed**, results can change between runs."
        ),
        code(
            "import random\n"
            "from sklearn.model_selection import train_test_split\n"
            "from sklearn.ensemble import RandomForestRegressor\n"
            "from sklearn.metrics import r2_score\n\n"
            "rng = np.random.default_rng(0)\n"
            "X_demo = rng.normal(size=(200, 3))\n"
            "y_demo = X_demo[:, 0] * 2 + rng.normal(0, 0.5, 200)\n\n"
            "def run_without_seed():\n"
            "    X_tr, X_te, y_tr, y_te = train_test_split(X_demo, y_demo, test_size=0.3)\n"
            "    m = RandomForestRegressor(n_estimators=50)  # no random_state\n"
            "    m.fit(X_tr, y_tr)\n"
            "    return r2_score(y_te, m.predict(X_te))\n\n"
            "unseeded = [run_without_seed() for _ in range(5)]\n"
            "print('R2 without fixed seed (5 runs):', np.round(unseeded, 4))\n\n"
            "SEED = 42\n"
            "np.random.seed(SEED)\n"
            "random.seed(SEED)\n\n"
            "def run_with_seed():\n"
            "    X_tr, X_te, y_tr, y_te = train_test_split(X_demo, y_demo, test_size=0.3, random_state=SEED)\n"
            "    m = RandomForestRegressor(n_estimators=50, random_state=SEED)\n"
            "    m.fit(X_tr, y_tr)\n"
            "    return r2_score(y_te, m.predict(X_te))\n\n"
            "seeded = [run_with_seed() for _ in range(5)]\n"
            "print('R2 with fixed seed (5 runs):', np.round(seeded, 4))\n\n"
            "try:\n"
            "    import torch\n"
            "    torch.manual_seed(SEED)\n"
            "    print('PyTorch seed set.')\n"
            "except ImportError:\n"
            "    print('PyTorch not installed — skip torch seed.')"
        ),
        md("## Section 5 — Reporting Results Clearly"),
        code(
            "metrics_table = pd.DataFrame([\n"
            "    {'Model': 'Seasonal Naive', 'MAE': 18.2, 'RMSE': 22.5, 'MAPE': 4.8},\n"
            "    {'Model': 'SARIMA', 'MAE': 12.1, 'RMSE': 15.3, 'MAPE': 3.1},\n"
            "    {'Model': 'Random Forest', 'MAE': 14.0, 'RMSE': 17.2, 'MAPE': 3.6},\n"
            "])\n"
            "display(metrics_table)\n\n"
            "cluster_table = pd.DataFrame([\n"
            "    {'Cluster': 0, 'Income': 'High', 'Spending': 'High', 'Visits': 'Frequent'},\n"
            "    {'Cluster': 1, 'Income': 'Low', 'Spending': 'Low', 'Visits': 'Rare'},\n"
            "    {'Cluster': 2, 'Income': 'Medium-High', 'Spending': 'Medium', 'Visits': 'Moderate'},\n"
            "])\n"
            "display(cluster_table)"
        ),
        md(
            "### Example clustering interpretation\n\n"
            "> The clustering analysis identified three customer groups. Cluster 0 contains high-value frequent customers, "
            "Cluster 1 contains low-frequency customers, and Cluster 2 contains moderate customers. "
            "The PCA visualization shows partial separation between clusters, but some overlap remains.\n\n"
            "### Example forecasting interpretation\n\n"
            "> The seasonal naive baseline achieved an RMSE of 22.5, while SARIMA achieved an RMSE of 15.3. "
            "Although SARIMA performed better, the improvement should be interpreted carefully because the test period is short."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.bar(metrics_table['Model'], metrics_table['RMSE'], color='steelblue')\n"
            "ax.set_title('Example RMSE comparison')\n"
            "ax.set_ylabel('RMSE')\n"
            "plt.xticks(rotation=20, ha='right')\n"
            "plt.tight_layout()\n"
            "plt.savefig(FIGURES_DIR / 'day3_example_metrics_bar.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md("## Section 6 — Combining Supervised and Unsupervised Workflows"),
        code(
            "from sklearn.cluster import KMeans\n"
            "from sklearn.preprocessing import StandardScaler\n"
            "from sklearn.linear_model import LinearRegression\n"
            "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\n"
            "if CUSTOMER_CSV.exists():\n"
            "    df = pd.read_csv(CUSTOMER_CSV)\n"
            "else:\n"
            "    from sklearn.datasets import make_blobs\n"
            "    X_b, _ = make_blobs(n_samples=300, centers=3, random_state=42)\n"
            "    df = pd.DataFrame(X_b, columns=['f1', 'f2', 'f3', 'f4'])\n\n"
            "feat = [c for c in df.columns if c not in ['customer_value', 'kmeans_cluster']][:4]\n"
            "X = df[feat].select_dtypes(include='number')\n"
            "scaler = StandardScaler()\n"
            "X_sc = scaler.fit_transform(X)\n"
            "df['cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_sc)\n\n"
            "rng = np.random.default_rng(42)\n"
            "df['customer_value'] = X.iloc[:, 0] * 100 + df['cluster'] * 50 + rng.normal(0, 20, len(df))\n\n"
            "y = df['customer_value']\n"
            "X_base = X\n"
            "X_plus = X.assign(cluster=df['cluster'])\n"
            "Xb_tr, Xb_te, y_tr, y_te = train_test_split(X_base, y, test_size=0.2, random_state=42)\n"
            "Xp_tr, Xp_te, _, _ = train_test_split(X_plus, y, test_size=0.2, random_state=42)\n\n"
            "m0 = LinearRegression().fit(Xb_tr, y_tr)\n"
            "m1 = LinearRegression().fit(Xp_tr, y_tr)\n"
            "cmp = pd.DataFrame([\n"
            "    {'Setup': 'Without cluster feature', 'MAE': mean_absolute_error(y_te, m0.predict(Xb_te)),\n"
            "     'RMSE': np.sqrt(mean_squared_error(y_te, m0.predict(Xb_te))), 'R2': r2_score(y_te, m0.predict(Xb_te))},\n"
            "    {'Setup': 'With cluster feature', 'MAE': mean_absolute_error(y_te, m1.predict(Xp_te)),\n"
            "     'RMSE': np.sqrt(mean_squared_error(y_te, m1.predict(Xp_te))), 'R2': r2_score(y_te, m1.predict(Xp_te))},\n"
            "])\n"
            "display(cmp.round(3))"
        ),
        md(
            "## Section 7 — Deployment Concept\n\n"
            "Deployment means making a trained model **usable** by others (API, batch job, app).\n\n"
            "- **Model artifact:** saved file (for example `.joblib`)\n"
            "- **API / batch job:** receives new data, returns predictions\n"
            "- **Hosting:** cloud, edge, or on-premise\n"
            "- **Testing:** validate before release"
        ),
        code(
            "import joblib\n"
            "from sklearn.linear_model import LinearRegression\n\n"
            "demo_X = np.array([[1.0, 2.0], [2.5, 3.1]])\n"
            "demo_y = np.array([3.0, 5.0])\n"
            "demo_model = LinearRegression().fit(demo_X, demo_y)\n\n"
            "model_path = DATA_DIR / 'demo_deployment_model.joblib'\n"
            "joblib.dump(demo_model, model_path)\n"
            "loaded_model = joblib.load(model_path)\n"
            "new_data = np.array([[1.5, 2.5]])\n"
            "print('Prediction on new data:', loaded_model.predict(new_data))"
        ),
        md(
            "## Section 8 — Model Maintenance\n\n"
            "Models can **decay** when real-world data changes (**data drift**).\n\n"
            "Responses: monitor metrics, inspect errors, retrain on recent data, version models, rollback if needed, "
            "respect privacy and security."
        ),
        code(
            "rng = np.random.default_rng(42)\n"
            "train_x = rng.normal(0, 1, 400)\n"
            "train_y = (train_x > 0).astype(int)\n"
            "test_x = rng.normal(1.2, 1.1, 400)  # shifted distribution\n"
            "test_y = (test_x > 0.2).astype(int)\n\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.metrics import accuracy_score\n\n"
            "clf = LogisticRegression(max_iter=1000)\n"
            "clf.fit(train_x.reshape(-1, 1), train_y)\n"
            "print('Accuracy on original-like test:', round(accuracy_score(test_y, clf.predict(test_x.reshape(-1, 1))), 3))\n\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "ax.hist(train_x, bins=25, alpha=0.6, label='Train distribution')\n"
            "ax.hist(test_x, bins=25, alpha=0.6, label='Shifted test distribution')\n"
            "ax.set_title('Simulated data drift')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.savefig(FIGURES_DIR / 'day3_data_drift.png', dpi=120, bbox_inches='tight')\n"
            "plt.show()"
        ),
        md(
            "## Section 9 — Final Project Draft Template\n\n"
            "Copy the template below into `reports/final_report.md`."
        ),
        md(
            "```markdown\n"
            "# Final Project Report\n\n"
            "## 1. Problem and Dataset\n"
            "Describe the problem, dataset, target variable, and goal.\n\n"
            "## 2. Data Cleaning\n"
            "Explain missing values, duplicates, outliers, and transformations.\n\n"
            "## 3. Feature Engineering\n"
            "Describe created features and why they are useful.\n\n"
            "## 4. Clustering Method and Parameters\n"
            "Explain K-Means or DBSCAN, parameter choices, and evaluation.\n\n"
            "## 5. PCA Visualization\n"
            "Show PC1/PC2 visualization and interpret cluster structure.\n\n"
            "## 6. Time-Based Split\n"
            "Explain train/test split and why random splitting was avoided.\n\n"
            "## 7. Baseline Forecast\n"
            "Describe naive, seasonal naive, or moving average baseline.\n\n"
            "## 8. Forecasting Models\n"
            "Compare at least one classical model and one ML model.\n\n"
            "## 9. Model Comparison\n"
            "Report MAE, RMSE, and other metrics.\n\n"
            "## 10. Interpretation and Limitations\n"
            "Explain results, assumptions, weaknesses, and practical meaning.\n\n"
            "## 11. Reproducibility\n"
            "Include code, random seed, package versions, environment, and GitHub link.\n"
            "```"
        ),
        md(
            "### Common mistakes / warnings\n"
            "- Reporting only accuracy without context\n"
            "- No seed / no `requirements.txt`\n"
            "- Test set used during feature engineering\n"
            "- Claiming causation from correlation or clusters"
        ),
        md(
            "## Section 10 — Final Student Checklist\n\n"
            "Submit:\n"
            "- Notebook 4 (this week's work)\n"
            "- Final project preparation draft\n"
            "- Plots and metrics table\n"
            "- Reproducibility information\n"
            "- GitHub or code link\n\n"
            "**Checklist**\n"
            "- [ ] Clustering completed and parameters explained\n"
            "- [ ] PCA visualization included\n"
            "- [ ] Time-based split used for forecasting\n"
            "- [ ] Baseline forecast implemented\n"
            "- [ ] At least two forecasting approaches compared\n"
            "- [ ] Results interpreted and limitations discussed\n"
            "- [ ] Random seed fixed\n"
            "- [ ] Environment documented\n"
            "- [ ] Code runs top to bottom"
        ),
        md(
            "## Mini summary\n"
            "- Communicate results clearly, not only model scores.\n"
            "- Organize projects so others can reproduce your work.\n"
            "- Treat clustering and forecasting as parts of one decision workflow.\n"
            "- Plan for deployment, monitoring, and drift after training."
        ),
        code("print('Week 4 Day 3 Reporting and Final Project Preparation completed successfully.')"),
    ]
    # Day 3 needs matplotlib import in early code cell - add imports cell
    cells.insert(
        2,
        code(
            "import sys\n"
            "import platform\n"
            "import random\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "%matplotlib inline"
        ),
    )
    write_notebook(LECTURE_DIR / "03_Week4_Day3_Reporting_Reproducibility_Final_Project.ipynb", cells)


def update_week4_readme():
    text = """# Week 4: Unsupervised Learning, Time Series, and Reporting

This week covers clustering, PCA, forecasting, and reproducible reporting for the final project.

## Lecture_Support Notebooks
- `01_Week4_Day1_Clustering_PCA.ipynb`
- `02_Week4_Day2_Time_Series_Forecasting.ipynb`
- `03_Week4_Day3_Reporting_Reproducibility_Final_Project.ipynb`

## Shared data (`data/week4/`)
- `week4_customer_segmentation.csv` (created by Day 1)
- `week4_air_passengers.csv` (created by Day 2)
- `week4_forecast_model_comparison.csv` (created by Day 2)
- `week4_environment_versions.csv` (created by Day 3)

## Figures (`figures/`)
Saved plots from the notebooks (feature distributions, PCA, forecasts, etc.).

## Suggested run order
1. Day 1 — Clustering + PCA
2. Day 2 — Time series forecasting (set `RUN_DEEP_LEARNING = False` if training is slow)
3. Day 3 — Reporting, reproducibility, final project template

## Optional packages
- `prophet` — Prophet forecasts (Day 2)
- `xgboost` — gradient boosting (Day 2 falls back to HistGradientBoosting)
- `torch` — LSTM, GRU, Transformer demos (Day 2)

Install extras: `pip install prophet xgboost torch`

## Regenerate notebooks
```bash
python scripts/build_week4_notebooks.py
```
"""
    (WEEK4_DIR / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    build_day1()
    build_day2()
    build_day3()
    update_week4_readme()
    print("Week 4 notebooks generated in Lecture_Support.")
