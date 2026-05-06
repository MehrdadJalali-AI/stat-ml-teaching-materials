from pathlib import Path
import nbformat as nbf


BASE = Path("/Users/mehrdadjalali/Documents/SRH-Courses/Modules/Statistics and Machine Learning/Hands-on/Statistics-and-Machine-Learning")
WEEK3_DIR = BASE / "Week_03_Classification_Evaluation_and_Regularization"
LECTURE_DIR = WEEK3_DIR / "Lecture_Support"
DATA_DIR = BASE / "data" / "week3"

LECTURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


COMMON_HELPER = """from pathlib import Path

def find_repo_root():
    cwd = Path.cwd().resolve()
    for candidate in [cwd] + list(cwd.parents):
        if (candidate / "requirements.txt").exists() and (candidate / "data").exists():
            return candidate
    return cwd

REPO_ROOT = find_repo_root()
DATASET_PATH = REPO_ROOT / "data" / "week3" / "week3_classification_sample.csv"
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
"""


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def build_day1():
    cells = []
    cells.append(nbf.v4.new_markdown_cell("# Week 3 Day 1 – Decision Trees and Evaluation\n\n## Learning objectives\n- Understand classification models and Decision Trees\n- Understand entropy, information gain, and tree splits\n- Compare different tree settings\n- Evaluate model performance using classification metrics"))
    cells.append(nbf.v4.new_markdown_cell("We use the breast cancer dataset from scikit-learn because it is beginner-friendly and binary classification is ideal for metric interpretation."))
    cells.append(nbf.v4.new_code_cell("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.tree import DecisionTreeClassifier, plot_tree\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n\n%matplotlib inline"))
    cells.append(nbf.v4.new_markdown_cell("### Shared path helper"))
    cells.append(nbf.v4.new_code_cell(COMMON_HELPER))
    cells.append(nbf.v4.new_markdown_cell("## Load dataset and save CSV copy"))
    cells.append(nbf.v4.new_code_cell("data = load_breast_cancer()\nfeature_names = list(data.feature_names)\ntarget_names = list(data.target_names)\n\ndf = pd.DataFrame(data.data, columns=feature_names)\ndf['target'] = data.target\n\ndf.to_csv(DATASET_PATH, index=False)\nprint('Saved dataset to:', DATASET_PATH)"))
    cells.append(nbf.v4.new_markdown_cell("## Basic exploration\n- `shape`: rows and columns\n- `head`: first rows\n- `value_counts`: class distribution\n- missing-value check with `isna().sum()`"))
    cells.append(nbf.v4.new_code_cell("print('Shape:', df.shape)\ndisplay(df.head())\nprint('Class distribution (0=malignant, 1=benign):')\nprint(df['target'].value_counts())\nprint('\\nFeature count:', len(feature_names))\nprint('Target labels:', target_names)\nprint('\\nMissing values per column:')\nprint(df.isna().sum().head())"))
    cells.append(nbf.v4.new_markdown_cell("## Train/test split\nWe split data so we can test generalization on unseen examples."))
    cells.append(nbf.v4.new_code_cell("X = df.drop(columns=['target'])\ny = df['target']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y\n)\nprint('Train shape:', X_train.shape)\nprint('Test shape:', X_test.shape)"))
    cells.append(nbf.v4.new_markdown_cell("## Helper function for evaluation metrics"))
    cells.append(nbf.v4.new_code_cell("def evaluate_classifier(name, model, X_train, X_test, y_train, y_test):\n    model.fit(X_train, y_train)\n    y_train_pred = model.predict(X_train)\n    y_test_pred = model.predict(X_test)\n\n    result = {\n        'Model': name,\n        'Train Accuracy': accuracy_score(y_train, y_train_pred),\n        'Test Accuracy': accuracy_score(y_test, y_test_pred),\n        'Precision': precision_score(y_test, y_test_pred),\n        'Recall': recall_score(y_test, y_test_pred),\n        'F1-score': f1_score(y_test, y_test_pred),\n        'Confusion Matrix': confusion_matrix(y_test, y_test_pred),\n        'Model Object': model,\n    }\n    return result"))
    cells.append(nbf.v4.new_markdown_cell("## Model 1: Default Decision Tree"))
    cells.append(nbf.v4.new_code_cell("m1 = evaluate_classifier('Default Tree', DecisionTreeClassifier(random_state=42), X_train, X_test, y_train, y_test)\nprint(m1['Confusion Matrix'])\nprint('Accuracy:', round(m1['Test Accuracy'], 3), 'Precision:', round(m1['Precision'], 3), 'Recall:', round(m1['Recall'], 3), 'F1:', round(m1['F1-score'], 3))"))
    cells.append(nbf.v4.new_markdown_cell("## Model 2: Gini criterion\nGini impurity measures class mixing in a node. Lower impurity means cleaner splits."))
    cells.append(nbf.v4.new_code_cell("m2 = evaluate_classifier('Gini depth=4', DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42), X_train, X_test, y_train, y_test)\nprint(m2['Confusion Matrix'])\nprint('Accuracy:', round(m2['Test Accuracy'], 3), 'Precision:', round(m2['Precision'], 3), 'Recall:', round(m2['Recall'], 3), 'F1:', round(m2['F1-score'], 3))"))
    cells.append(nbf.v4.new_markdown_cell("## Model 3: Entropy criterion\nEntropy is used in ID3-style intuition. Information gain chooses splits that reduce entropy most."))
    cells.append(nbf.v4.new_code_cell("m3 = evaluate_classifier('Entropy depth=4', DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42), X_train, X_test, y_train, y_test)\nprint(m3['Confusion Matrix'])\nprint('Accuracy:', round(m3['Test Accuracy'], 3), 'Precision:', round(m3['Precision'], 3), 'Recall:', round(m3['Recall'], 3), 'F1:', round(m3['F1-score'], 3))"))
    cells.append(nbf.v4.new_markdown_cell("## Model 4: Compare depth settings\nDeep trees can overfit: training performance goes up, but test performance may stop improving."))
    cells.append(nbf.v4.new_code_cell("depths = [1, 2, 3, 4, 5, None]\ndepth_rows = []\n\nfor d in depths:\n    model = DecisionTreeClassifier(max_depth=d, random_state=42)\n    model.fit(X_train, y_train)\n    train_acc = accuracy_score(y_train, model.predict(X_train))\n    test_acc = accuracy_score(y_test, model.predict(X_test))\n    depth_rows.append({'max_depth': str(d), 'train_accuracy': train_acc, 'test_accuracy': test_acc})\n\ndepth_df = pd.DataFrame(depth_rows)\ndisplay(depth_df)"))
    cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(8,4))\nplt.plot(depth_df['max_depth'], depth_df['train_accuracy'], marker='o', label='Train accuracy')\nplt.plot(depth_df['max_depth'], depth_df['test_accuracy'], marker='o', label='Test accuracy')\nplt.title('Decision Tree Depth vs Accuracy')\nplt.xlabel('max_depth')\nplt.ylabel('Accuracy')\nplt.legend()\nplt.grid(alpha=0.3)\nplt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## Feature importance from best test-accuracy tree"))
    cells.append(nbf.v4.new_code_cell("best_depth = depth_df.sort_values('test_accuracy', ascending=False).iloc[0]['max_depth']\nbest_depth_param = None if best_depth == 'None' else int(best_depth)\nbest_tree = DecisionTreeClassifier(max_depth=best_depth_param, random_state=42)\nbest_tree.fit(X_train, y_train)\n\nfi = pd.DataFrame({'feature': feature_names, 'importance': best_tree.feature_importances_}).sort_values('importance', ascending=False)\ndisplay(fi.head(10))\n\nplt.figure(figsize=(9,4))\nplt.bar(fi.head(10)['feature'], fi.head(10)['importance'])\nplt.title('Top 10 Feature Importances')\nplt.xticks(rotation=45, ha='right')\nplt.tight_layout()\nplt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## Visualize a small tree"))
    cells.append(nbf.v4.new_code_cell("small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)\nsmall_tree.fit(X_train, y_train)\n\nplt.figure(figsize=(16, 8))\nplot_tree(small_tree, feature_names=feature_names, class_names=target_names, filled=True, max_depth=2, fontsize=8)\nplt.title('Decision Tree Visualization (truncated depth for readability)')\nplt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## Summary table for compared tree models"))
    cells.append(nbf.v4.new_code_cell("summary = pd.DataFrame([\n    {'Model': m1['Model'], 'Criterion': 'default', 'max_depth': 'default', 'Train Accuracy': m1['Train Accuracy'], 'Test Accuracy': m1['Test Accuracy'], 'Precision': m1['Precision'], 'Recall': m1['Recall'], 'F1-score': m1['F1-score']},\n    {'Model': m2['Model'], 'Criterion': 'gini', 'max_depth': 4, 'Train Accuracy': m2['Train Accuracy'], 'Test Accuracy': m2['Test Accuracy'], 'Precision': m2['Precision'], 'Recall': m2['Recall'], 'F1-score': m2['F1-score']},\n    {'Model': m3['Model'], 'Criterion': 'entropy', 'max_depth': 4, 'Train Accuracy': m3['Train Accuracy'], 'Test Accuracy': m3['Test Accuracy'], 'Precision': m3['Precision'], 'Recall': m3['Recall'], 'F1-score': m3['F1-score']},\n]).sort_values('Test Accuracy', ascending=False)\ndisplay(summary.round(3))"))
    cells.append(nbf.v4.new_markdown_cell("## Student reflection\n- Which tree performed best?\n- Did the deepest tree generalize best?\n- Why might a simpler tree be better?\n- Which metric matters most for this task?"))
    cells.append(nbf.v4.new_code_cell("print('Week 3 Day 1 Decision Trees and Evaluation completed successfully.')"))
    write_notebook(LECTURE_DIR / "01_Week3_Day1_Decision_Trees_and_Evaluation.ipynb", cells)


def build_day2():
    cells = []
    cells.append(nbf.v4.new_markdown_cell("# Week 3 Day 2 – KNN, Best k, and Model Comparison\n\n## Learning objectives\n- Understand KNN intuition and distance-based classification\n- See why scaling matters for KNN\n- Find a good `k`\n- Compare KNN with Decision Tree"))
    cells.append(nbf.v4.new_code_cell("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom sklearn.datasets import load_breast_cancer\nfrom sklearn.model_selection import train_test_split, cross_val_score\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n\n%matplotlib inline"))
    cells.append(nbf.v4.new_code_cell(COMMON_HELPER))
    cells.append(nbf.v4.new_markdown_cell("## Load same dataset as Day 1"))
    cells.append(nbf.v4.new_code_cell("if DATASET_PATH.exists():\n    df = pd.read_csv(DATASET_PATH)\nelse:\n    data = load_breast_cancer()\n    df = pd.DataFrame(data.data, columns=data.feature_names)\n    df['target'] = data.target\n    df.to_csv(DATASET_PATH, index=False)\n\ndisplay(df.head())"))
    cells.append(nbf.v4.new_markdown_cell("## Split data"))
    cells.append(nbf.v4.new_code_cell("X = df.drop(columns=['target'])\ny = df['target']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\nprint(X_train.shape, X_test.shape)"))
    cells.append(nbf.v4.new_markdown_cell("## Scaling for KNN\nKNN uses distance, so scaling is important to avoid one feature dominating others."))
    cells.append(nbf.v4.new_code_cell("scaler = StandardScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)"))
    cells.append(nbf.v4.new_markdown_cell("## KNN with one fixed k"))
    cells.append(nbf.v4.new_code_cell("knn5 = KNeighborsClassifier(n_neighbors=5)\nknn5.fit(X_train_scaled, y_train)\ny_pred_knn5 = knn5.predict(X_test_scaled)\n\nprint('Confusion Matrix:\\n', confusion_matrix(y_test, y_pred_knn5))\nprint('Accuracy:', round(accuracy_score(y_test, y_pred_knn5), 3))\nprint('Precision:', round(precision_score(y_test, y_pred_knn5), 3))\nprint('Recall:', round(recall_score(y_test, y_pred_knn5), 3))\nprint('F1:', round(f1_score(y_test, y_pred_knn5), 3))"))
    cells.append(nbf.v4.new_markdown_cell("## Compare KNN with and without scaling"))
    cells.append(nbf.v4.new_code_cell("knn_unscaled = KNeighborsClassifier(n_neighbors=5)\nknn_unscaled.fit(X_train, y_train)\ny_pred_unscaled = knn_unscaled.predict(X_test)\n\nscale_cmp = pd.DataFrame([\n    {'Model': 'KNN k=5 (unscaled)', 'Accuracy': accuracy_score(y_test, y_pred_unscaled), 'Precision': precision_score(y_test, y_pred_unscaled), 'Recall': recall_score(y_test, y_pred_unscaled), 'F1': f1_score(y_test, y_pred_unscaled)},\n    {'Model': 'KNN k=5 (scaled)', 'Accuracy': accuracy_score(y_test, y_pred_knn5), 'Precision': precision_score(y_test, y_pred_knn5), 'Recall': recall_score(y_test, y_pred_knn5), 'F1': f1_score(y_test, y_pred_knn5)},\n])\ndisplay(scale_cmp.round(3))"))
    cells.append(nbf.v4.new_markdown_cell("## Find best k (1 to 30)"))
    cells.append(nbf.v4.new_code_cell("rows = []\nfor k in range(1, 31):\n    knn = KNeighborsClassifier(n_neighbors=k)\n    knn.fit(X_train_scaled, y_train)\n    y_train_pred = knn.predict(X_train_scaled)\n    y_test_pred = knn.predict(X_test_scaled)\n    rows.append({\n        'k': k,\n        'train_accuracy': accuracy_score(y_train, y_train_pred),\n        'test_accuracy': accuracy_score(y_test, y_test_pred),\n        'test_f1': f1_score(y_test, y_test_pred),\n    })\n\nk_df = pd.DataFrame(rows)\ndisplay(k_df.head())"))
    cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(10,4))\nplt.plot(k_df['k'], k_df['train_accuracy'], label='Train accuracy')\nplt.plot(k_df['k'], k_df['test_accuracy'], label='Test accuracy')\nplt.title('k vs Accuracy (KNN)')\nplt.xlabel('k')\nplt.ylabel('Accuracy')\nplt.legend(); plt.grid(alpha=0.3); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("Small `k` can overfit. Very large `k` can underfit. We choose `k` that generalizes well on test/CV."))
    cells.append(nbf.v4.new_markdown_cell("## Optional cross-validation for k selection"))
    cells.append(nbf.v4.new_code_cell("cv_rows = []\nfor k in range(1, 31):\n    knn = KNeighborsClassifier(n_neighbors=k)\n    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')\n    cv_rows.append({'k': k, 'cv_mean_accuracy': scores.mean()})\ncv_df = pd.DataFrame(cv_rows)\n\nplt.figure(figsize=(10,4))\nplt.plot(cv_df['k'], cv_df['cv_mean_accuracy'])\nplt.title('k vs Mean CV Accuracy')\nplt.xlabel('k')\nplt.ylabel('Mean CV accuracy')\nplt.grid(alpha=0.3)\nplt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## Train best KNN model"))
    cells.append(nbf.v4.new_code_cell("best_k = int(cv_df.sort_values('cv_mean_accuracy', ascending=False).iloc[0]['k'])\nbest_knn = KNeighborsClassifier(n_neighbors=best_k)\nbest_knn.fit(X_train_scaled, y_train)\ny_pred_best_knn = best_knn.predict(X_test_scaled)\n\nprint('Best k:', best_k)\nprint('Confusion matrix:\\n', confusion_matrix(y_test, y_pred_best_knn))\nprint('\\nClassification report:\\n', classification_report(y_test, y_pred_best_knn))"))
    cells.append(nbf.v4.new_markdown_cell("## Compare best KNN with Decision Tree"))
    cells.append(nbf.v4.new_code_cell("tree = DecisionTreeClassifier(max_depth=4, random_state=42)\ntree.fit(X_train, y_train)\ny_pred_tree = tree.predict(X_test)\n\ncmp = pd.DataFrame([\n    {'Model': f'KNN (k={best_k})', 'Accuracy': accuracy_score(y_test, y_pred_best_knn), 'Precision': precision_score(y_test, y_pred_best_knn), 'Recall': recall_score(y_test, y_pred_best_knn), 'F1': f1_score(y_test, y_pred_best_knn)},\n    {'Model': 'Decision Tree (max_depth=4)', 'Accuracy': accuracy_score(y_test, y_pred_tree), 'Precision': precision_score(y_test, y_pred_tree), 'Recall': recall_score(y_test, y_pred_tree), 'F1': f1_score(y_test, y_pred_tree)},\n])\ndisplay(cmp.round(3))"))
    cells.append(nbf.v4.new_code_cell("metrics = ['Accuracy', 'Precision', 'Recall', 'F1']\nx = np.arange(len(metrics))\nwidth = 0.35\n\nplt.figure(figsize=(8,4))\nplt.bar(x - width/2, cmp.loc[0, metrics], width, label=cmp.loc[0, 'Model'])\nplt.bar(x + width/2, cmp.loc[1, metrics], width, label=cmp.loc[1, 'Model'])\nplt.xticks(x, metrics)\nplt.ylim(0, 1.05)\nplt.title('Model Comparison by Metric')\nplt.legend()\nplt.tight_layout()\nplt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## Interpretation\n- Which model has higher recall?\n- Which has higher precision?\n- Which errors are more important in this task?"))
    cells.append(nbf.v4.new_markdown_cell("## Student tasks\n1. Try `metric='manhattan'` in KNN.\n2. Compare with Euclidean metric.\n3. Test another `k`.\n4. Explain whether KNN or Decision Tree is easier to interpret."))
    cells.append(nbf.v4.new_code_cell("print('Week 3 Day 2 KNN and Model Comparison completed successfully.')"))
    write_notebook(LECTURE_DIR / "02_Week3_Day2_KNN_Best_k_and_Model_Comparison.ipynb", cells)


def build_day3():
    cells = []
    cells.append(nbf.v4.new_markdown_cell("# Week 3 Day 3 – Concept Drift, Model Selection, ROC–AUC\n\n## Learning objectives\n- Understand concept drift\n- Compare multiple classification models\n- Understand ROC and AUC\n- See threshold trade-offs\n- Build hyperparameter-tuning mindset"))
    cells.append(nbf.v4.new_code_cell("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix\n\n%matplotlib inline"))
    cells.append(nbf.v4.new_code_cell(COMMON_HELPER))
    cells.append(nbf.v4.new_markdown_cell("## Concept drift synthetic example"))
    cells.append(nbf.v4.new_code_cell("rng = np.random.default_rng(42)\n\n# Period 1\nn1 = 300\nx1_p1 = rng.normal(0, 1, n1)\nx2_p1 = rng.normal(0, 1, n1)\ny_p1 = ((x1_p1 + 0.6 * x2_p1 + rng.normal(0, 0.5, n1)) > 0).astype(int)\n\n# Period 2 (drift: feature influence shifts)\nn2 = 300\nx1_p2 = rng.normal(0.2, 1.1, n2)\nx2_p2 = rng.normal(0.4, 1.1, n2)\ny_p2 = ((0.2 * x1_p2 + 1.1 * x2_p2 + rng.normal(0, 0.6, n2)) > 0.3).astype(int)\n\nperiod1 = pd.DataFrame({'feature_1': x1_p1, 'feature_2': x2_p1, 'target': y_p1})\nperiod2 = pd.DataFrame({'feature_1': x1_p2, 'feature_2': x2_p2, 'target': y_p2})"))
    cells.append(nbf.v4.new_markdown_cell("## Visualize drift"))
    cells.append(nbf.v4.new_code_cell("fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)\naxes[0].scatter(period1['feature_1'], period1['feature_2'], c=period1['target'], alpha=0.7)\naxes[0].set_title('Period 1')\naxes[1].scatter(period2['feature_1'], period2['feature_2'], c=period2['target'], alpha=0.7)\naxes[1].set_title('Period 2 (drift)')\nfor ax in axes:\n    ax.set_xlabel('feature_1')\n    ax.set_ylabel('feature_2')\nplt.tight_layout(); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## Train on Period 1, evaluate on Period 1 and Period 2"))
    cells.append(nbf.v4.new_code_cell("X1 = period1[['feature_1', 'feature_2']]\ny1 = period1['target']\nX2 = period2[['feature_1', 'feature_2']]\ny2 = period2['target']\n\nX1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=42, stratify=y1)\n\ndrift_model = LogisticRegression(max_iter=1000)\ndrift_model.fit(X1_train, y1_train)\n\npred_p1 = drift_model.predict(X1_test)\npred_p2 = drift_model.predict(X2)\n\ndrift_cmp = pd.DataFrame([\n    {'Evaluation period': 'Period 1 test', 'Accuracy': accuracy_score(y1_test, pred_p1), 'Precision': precision_score(y1_test, pred_p1), 'Recall': recall_score(y1_test, pred_p1), 'F1': f1_score(y1_test, pred_p1)},\n    {'Evaluation period': 'Period 2 full', 'Accuracy': accuracy_score(y2, pred_p2), 'Precision': precision_score(y2, pred_p2), 'Recall': recall_score(y2, pred_p2), 'F1': f1_score(y2, pred_p2)},\n])\ndisplay(drift_cmp.round(3))"))
    cells.append(nbf.v4.new_markdown_cell("If performance drops on new data, this may indicate concept drift.\n\nSimple responses:\n- monitor performance continuously\n- retrain on recent data\n- use sliding windows\n- compare old and new models"))
    cells.append(nbf.v4.new_markdown_cell("## Model selection on real dataset (Week 3 shared dataset)"))
    cells.append(nbf.v4.new_code_cell("if DATASET_PATH.exists():\n    df = pd.read_csv(DATASET_PATH)\nelse:\n    from sklearn.datasets import load_breast_cancer\n    d = load_breast_cancer()\n    df = pd.DataFrame(d.data, columns=d.feature_names)\n    df['target'] = d.target\n    df.to_csv(DATASET_PATH, index=False)\n\nX = df.drop(columns=['target'])\ny = df['target']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)"))
    cells.append(nbf.v4.new_code_cell("models = {\n    'Logistic Regression': LogisticRegression(max_iter=1000),\n    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),\n    'KNN': KNeighborsClassifier(n_neighbors=7),\n}\n\nrows = []\nproba = {}\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    pred = model.predict(X_test)\n    rows.append({\n        'Model': name,\n        'Accuracy': accuracy_score(y_test, pred),\n        'Precision': precision_score(y_test, pred),\n        'Recall': recall_score(y_test, pred),\n        'F1': f1_score(y_test, pred),\n    })\n    proba[name] = model.predict_proba(X_test)[:, 1]\n\nmodel_cmp = pd.DataFrame(rows)\n\na_rows = []\nfor name in models:\n    auc = roc_auc_score(y_test, proba[name])\n    a_rows.append({'Model': name, 'ROC-AUC': auc})\nauc_df = pd.DataFrame(a_rows)\n\nfinal_cmp = model_cmp.merge(auc_df, on='Model')\ndisplay(final_cmp.round(3))"))
    cells.append(nbf.v4.new_markdown_cell("Model selection depends on performance metrics, interpretability, speed, and real-world error costs."))
    cells.append(nbf.v4.new_markdown_cell("## ROC curves and AUC"))
    cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(7, 6))\nfor name in models:\n    fpr, tpr, _ = roc_curve(y_test, proba[name])\n    auc = roc_auc_score(y_test, proba[name])\n    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')\n\nplt.plot([0, 1], [0, 1], 'k--', label='Random baseline')\nplt.xlabel('False Positive Rate')\nplt.ylabel('True Positive Rate / Recall')\nplt.title('ROC Curves')\nplt.legend()\nplt.grid(alpha=0.3)\nplt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## Threshold analysis (Logistic Regression)\nLower threshold often raises recall, higher threshold often raises precision."))
    cells.append(nbf.v4.new_code_cell("log_model = models['Logistic Regression']\nlog_proba = proba['Logistic Regression']\n\nthr_rows = []\nfor thr in [0.3, 0.5, 0.7]:\n    pred_thr = (log_proba >= thr).astype(int)\n    tn, fp, fn, tp = confusion_matrix(y_test, pred_thr).ravel()\n    thr_rows.append({\n        'Threshold': thr,\n        'Precision': precision_score(y_test, pred_thr),\n        'Recall': recall_score(y_test, pred_thr),\n        'F1': f1_score(y_test, pred_thr),\n        'False Positives': fp,\n        'False Negatives': fn,\n    })\n\nthr_df = pd.DataFrame(thr_rows)\ndisplay(thr_df.round(3))"))
    cells.append(nbf.v4.new_markdown_cell("## Hyperparameter tuning mindset\n- KNN: `n_neighbors`\n- Decision Tree: `max_depth`\n- Logistic Regression: `C`\n\nTune on validation/CV, not on final test set."))
    cells.append(nbf.v4.new_markdown_cell("## Mini GridSearch examples"))
    cells.append(nbf.v4.new_code_cell("knn_grid = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3,5,7,9,11]}, cv=5, scoring='f1')\nknn_grid.fit(X_train, y_train)\n\ndt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), {'max_depth': [2,3,4,5,6,None]}, cv=5, scoring='f1')\ndt_grid.fit(X_train, y_train)\n\nprint('Best KNN params:', knn_grid.best_params_, '| best CV F1:', round(knn_grid.best_score_, 3))\nprint('Best Tree params:', dt_grid.best_params_, '| best CV F1:', round(dt_grid.best_score_, 3))"))
    cells.append(nbf.v4.new_markdown_cell("## Final model comparison table"))
    cells.append(nbf.v4.new_code_cell("best_models = {\n    'Logistic Regression': LogisticRegression(max_iter=1000),\n    'Decision Tree': dt_grid.best_estimator_,\n    'KNN': knn_grid.best_estimator_,\n}\n\nrows = []\nfor name, model in best_models.items():\n    model.fit(X_train, y_train)\n    pred = model.predict(X_test)\n    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])\n    rows.append({\n        'Model': name,\n        'Best hyperparameters': str(model.get_params()),\n        'Accuracy': accuracy_score(y_test, pred),\n        'Precision': precision_score(y_test, pred),\n        'Recall': recall_score(y_test, pred),\n        'F1': f1_score(y_test, pred),\n        'ROC-AUC': auc,\n    })\n\nfinal_table = pd.DataFrame(rows)\ndisplay(final_table[['Model','Accuracy','Precision','Recall','F1','ROC-AUC']].round(3))"))
    cells.append(nbf.v4.new_markdown_cell("## Final interpretation\nChoose model based on metric priorities, error costs, and interpretability.\n\nQuestions:\n- Which model would you choose and why?\n- Which metric supports your choice?\n- Is there any overfitting signal?\n- Would it still work if data drifts?"))
    cells.append(nbf.v4.new_markdown_cell("## Student tasks\n1. Change threshold and compare precision/recall.\n2. Try another hyperparameter value.\n3. Explain ROC-AUC in your own words.\n4. Give one real-world concept drift example.\n5. Write a non-technical summary of your chosen model."))
    cells.append(nbf.v4.new_code_cell("print('Week 3 Day 3 Concept Drift, Model Selection, ROC-AUC completed successfully.')"))
    write_notebook(LECTURE_DIR / "03_Week3_Day3_Concept_Drift_Model_Selection_ROC_AUC.ipynb", cells)


def update_week3_readme():
    text = """# Week 3: Classification, Evaluation, and Regularization

This week focuses on beginner-friendly classification workflows.

## Lecture_Support Notebooks
- `01_Week3_Day1_Decision_Trees_and_Evaluation.ipynb`
- `02_Week3_Day2_KNN_Best_k_and_Model_Comparison.ipynb`
- `03_Week3_Day3_Concept_Drift_Model_Selection_ROC_AUC.ipynb`

## Shared data
- `data/week3/week3_classification_sample.csv`

## Suggested run order
1. Day 1 (Decision Trees)
2. Day 2 (KNN and comparison)
3. Day 3 (Concept drift, ROC-AUC, model selection)
"""
    (WEEK3_DIR / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    build_day1()
    build_day2()
    build_day3()
    update_week3_readme()
    print("Week 3 notebooks generated in Lecture_Support.")
