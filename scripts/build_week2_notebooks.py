from pathlib import Path
import nbformat as nbf


BASE = Path("/Users/mehrdadjalali/Documents/SRH-Courses/Modules/Statistics and Machine Learning/Hands-on/Statistics-and-Machine-Learning")
NB_DIR = BASE / "notebooks" / "week2"
DATA_DIR = BASE / "data" / "week2"
NB_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

COMMON_HELPER_MD = """### Shared dataset helper
To keep all three days connected, we use the same dataset path and generation logic.
- Day 1 creates and saves the dataset.
- Day 2 and Day 3 load it and regenerate it only if missing.
"""

COMMON_HELPER_CODE = """from pathlib import Path

def find_repo_root():
    cwd = Path.cwd().resolve()
    for candidate in [cwd] + list(cwd.parents):
        if (candidate / "requirements.txt").exists() and (candidate / "data").exists():
            return candidate
    return cwd

REPO_ROOT = find_repo_root()
DATASET_PATH = REPO_ROOT / "data" / "week2" / "house_energy_week2_sample.csv"
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

def create_house_energy_dataset(path, n_rows=150, seed=42):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    house_size_m2 = np.clip(rng.normal(135, 48, n_rows), 40, 460)
    bedrooms = np.clip(np.round(house_size_m2 / 35 + rng.normal(0, 1, n_rows)), 1, 8).astype(int)
    building_age_years = np.clip(rng.normal(32, 22, n_rows), 1, 120)
    city = rng.choice(["Berlin", "Munich", "Hamburg", "Cologne", "Leipzig"], size=n_rows, p=[0.25, 0.2, 0.2, 0.2, 0.15])
    heating_type = rng.choice(["Gas", "Electric", "Heat Pump", "District"], size=n_rows, p=[0.4, 0.25, 0.2, 0.15])
    insulation_grade = rng.choice(["A", "B", "C", "D"], size=n_rows, p=[0.2, 0.35, 0.3, 0.15])
    average_temperature_c = np.clip(rng.normal(10.5, 5, n_rows), -8, 24)

    insulation_effect = pd.Series(insulation_grade).map({"A": -1300, "B": -650, "C": 200, "D": 1050}).to_numpy()
    heating_effect = pd.Series(heating_type).map({"Heat Pump": -900, "District": -300, "Gas": 220, "Electric": 520}).to_numpy()
    city_effect = pd.Series(city).map({"Berlin": 120, "Munich": 270, "Hamburg": 170, "Cologne": 100, "Leipzig": -40}).to_numpy()
    noise = rng.normal(0, 450, n_rows)

    energy_consumption_kwh = (
        3500
        + 23 * house_size_m2
        + 15 * building_age_years
        + (14 - average_temperature_c) * 175
        + insulation_effect
        + heating_effect
        + city_effect
        + noise
    )
    energy_consumption_kwh = np.clip(energy_consumption_kwh, 1500, None)

    df = pd.DataFrame({
        "house_size_m2": house_size_m2.round(1),
        "bedrooms": bedrooms,
        "building_age_years": building_age_years.round(1),
        "city": city,
        "heating_type": heating_type,
        "insulation_grade": insulation_grade,
        "average_temperature_c": average_temperature_c.round(1),
        "energy_consumption_kwh": energy_consumption_kwh.round(1),
    })

    for col in ["house_size_m2", "building_age_years", "average_temperature_c"]:
        idx = rng.choice(df.index, size=6, replace=False)
        df.loc[idx, col] = np.nan
    for col in ["city", "heating_type", "insulation_grade"]:
        idx = rng.choice(df.index, size=4, replace=False)
        df.loc[idx, col] = np.nan

    df.loc[4, ["house_size_m2", "energy_consumption_kwh"]] = [520, 35000]
    df.loc[58, ["building_age_years", "energy_consumption_kwh"]] = [145, 29000]
    df.loc[97, ["average_temperature_c", "energy_consumption_kwh"]] = [-20, 30500]

    duplicates = df.iloc[[12, 35]].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    df.to_csv(path, index=False)
    return df
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
    cells.append(nbf.v4.new_markdown_cell("# Week 2 – Day 1: Dataset Readiness, Preprocessing & Feature Engineering\n\n## 1) Introduction\nDay 1 is about making raw data ready for machine learning. We focus on preprocessing, data quality, and careful feature engineering before model training."))
    cells.append(nbf.v4.new_markdown_cell("## 2) Import libraries\nWe use `pandas`, `numpy`, and `matplotlib` for data handling and visualization, plus scikit-learn scaling tools."))
    cells.append(nbf.v4.new_code_cell("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom sklearn.preprocessing import StandardScaler, MinMaxScaler\n\n%matplotlib inline"))
    cells.append(nbf.v4.new_markdown_cell("- `pandas`: table data operations.\n- `numpy`: numerical operations and reproducible random generation.\n- `matplotlib.pyplot`: plots for outliers and smoothing.\n- `StandardScaler`: standardization.\n- `MinMaxScaler`: normalization to [0,1]."))
    cells.append(nbf.v4.new_markdown_cell(COMMON_HELPER_MD))
    cells.append(nbf.v4.new_code_cell(COMMON_HELPER_CODE))
    cells.append(nbf.v4.new_markdown_cell("## 3) Create and save the sample dataset\nGenerate and save the shared Week 2 dataset to `data/week2/house_energy_week2_sample.csv`."))
    cells.append(nbf.v4.new_code_cell("df_generated = create_house_energy_dataset(DATASET_PATH, n_rows=150, seed=42)\nprint('Saved dataset to:', DATASET_PATH)\nprint('Generated shape:', df_generated.shape)"))
    cells.append(nbf.v4.new_markdown_cell("## 4) Load the dataset\nUsing `pd.read_csv()` mirrors real project workflow: save data, then load and explore it."))
    cells.append(nbf.v4.new_code_cell("df = pd.read_csv(DATASET_PATH)\ndisplay(df.head())"))
    cells.append(nbf.v4.new_markdown_cell("## 5) First look at the data\n- `df.head()`: first rows preview.\n- `df.shape`: dataset size.\n- `df.info()`: dtypes and non-null counts.\n- `df.describe()`: summary stats.\n- `df.dtypes`: each column's type.\n- `df.columns`: list of columns."))
    cells.append(nbf.v4.new_code_cell("display(df.head())\nprint('Shape:', df.shape)\nprint('\\nInfo:')\ndf.info()\nprint('\\nDescribe:')\ndisplay(df.describe())\nprint('\\nDtypes:')\nprint(df.dtypes)\nprint('\\nColumns:')\nprint(df.columns.tolist())"))
    cells.append(nbf.v4.new_markdown_cell("## 6) Identify target and features\nTarget is what we predict. Features are inputs used for prediction."))
    cells.append(nbf.v4.new_code_cell("target = 'energy_consumption_kwh'\nX = df.drop(columns=[target])\ny = df[target]\n\nnumerical_features = ['house_size_m2', 'bedrooms', 'building_age_years', 'average_temperature_c']\ncategorical_features = ['city', 'heating_type', 'insulation_grade']\n\nprint('Target:', target)\nprint('X shape:', X.shape, '| y shape:', y.shape)\nprint('Numerical features:', numerical_features)\nprint('Categorical features:', categorical_features)"))
    cells.append(nbf.v4.new_markdown_cell("## 7) Check missing values\n`isna().sum()` gives counts, and `isna().mean()*100` gives percentages."))
    cells.append(nbf.v4.new_code_cell("missing_count = df.isna().sum()\nmissing_pct = df.isna().mean() * 100\n\ndisplay(missing_count)\ndisplay(missing_pct.round(2))"))
    cells.append(nbf.v4.new_markdown_cell("## 8) Handle missing values manually\n- Mean: arithmetic average.\n- Median: middle value (robust to outliers).\n- Mode: most frequent value (good for categorical data)."))
    cells.append(nbf.v4.new_code_cell("df_manual = df.copy()\nfor col in numerical_features:\n    df_manual[col] = df_manual[col].fillna(df_manual[col].median())\nfor col in categorical_features:\n    df_manual[col] = df_manual[col].fillna(df_manual[col].mode()[0])\nprint(df_manual.isna().sum())"))
    cells.append(nbf.v4.new_markdown_cell("## 9) Check duplicates\nUse `duplicated()` to inspect repeated rows and `drop_duplicates()` to remove accidental duplicates."))
    cells.append(nbf.v4.new_code_cell("print('Duplicate rows:', df_manual.duplicated().sum())\ndisplay(df_manual[df_manual.duplicated()])\ndf_no_dup = df_manual.drop_duplicates().copy()\nprint('Shape after drop_duplicates:', df_no_dup.shape)"))
    cells.append(nbf.v4.new_markdown_cell("## 10) Inspect outliers\nWe inspect with summary stats, boxplots, and IQR flags. Not all outliers should be removed automatically."))
    cells.append(nbf.v4.new_code_cell("display(df_no_dup[['house_size_m2','building_age_years','energy_consumption_kwh']].describe())\n\nfig, axes = plt.subplots(1, 3, figsize=(15, 4))\naxes[0].boxplot(df_no_dup['house_size_m2']); axes[0].set_title('house_size_m2')\naxes[1].boxplot(df_no_dup['building_age_years']); axes[1].set_title('building_age_years')\naxes[2].boxplot(df_no_dup['energy_consumption_kwh']); axes[2].set_title('energy_consumption_kwh')\nplt.tight_layout(); plt.show()\n\ndef iqr_flags(dataframe, col):\n    q1 = dataframe[col].quantile(0.25)\n    q3 = dataframe[col].quantile(0.75)\n    iqr = q3 - q1\n    lower = q1 - 1.5 * iqr\n    upper = q3 + 1.5 * iqr\n    flags = (dataframe[col] < lower) | (dataframe[col] > upper)\n    return flags, lower, upper\n\nfor col in ['house_size_m2','building_age_years','energy_consumption_kwh']:\n    flags, lower, upper = iqr_flags(df_no_dup, col)\n    print(f'{col}: {flags.sum()} possible outliers | [{lower:.2f}, {upper:.2f}]')"))
    cells.append(nbf.v4.new_markdown_cell("## 11) Noise and smoothing\nRolling mean smooths short-term variation and helps reveal trend direction."))
    cells.append(nbf.v4.new_code_cell("df_noise = df_no_dup.copy()\ndf_noise['energy_smoothed'] = df_noise['energy_consumption_kwh'].rolling(window=5, min_periods=1).mean()\n\nplt.figure(figsize=(12,5))\nplt.plot(df_noise.index, df_noise['energy_consumption_kwh'], label='Original', alpha=0.7)\nplt.plot(df_noise.index, df_noise['energy_smoothed'], label='Smoothed', linewidth=2)\nplt.title('Original vs Smoothed Energy Consumption')\nplt.xlabel('Row index'); plt.ylabel('energy_consumption_kwh')\nplt.legend(); plt.grid(alpha=0.3); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## 12) Encoding categorical variables\n`pd.get_dummies()` performs one-hot encoding so categories become numeric columns."))
    cells.append(nbf.v4.new_code_cell("encoded = pd.get_dummies(df_no_dup[categorical_features], drop_first=False)\nprint(encoded.shape)\ndisplay(encoded.head())"))
    cells.append(nbf.v4.new_markdown_cell("## 13) Feature scaling\nFormulas:\n- `z = (x - mean) / standard deviation`\n- `x_scaled = (x - x_min) / (x_max - x_min)`"))
    cells.append(nbf.v4.new_code_cell("num_data = df_no_dup[numerical_features].copy().fillna(df_no_dup[numerical_features].median())\nstd = StandardScaler()\nmm = MinMaxScaler()\nstd_df = pd.DataFrame(std.fit_transform(num_data), columns=[f'{c}_std' for c in numerical_features])\nmm_df = pd.DataFrame(mm.fit_transform(num_data), columns=[f'{c}_mm' for c in numerical_features])\n\ndisplay(num_data.describe().round(2))\ndisplay(std_df.describe().round(2))\ndisplay(mm_df.describe().round(2))"))
    cells.append(nbf.v4.new_markdown_cell("## 14) Basic feature engineering\nWe add four features for exploration.\n\n`energy_per_m2` uses target information, so it is useful for exploration but should **not** be used as a model input feature (leakage risk)."))
    cells.append(nbf.v4.new_code_cell("df_feat = df_no_dup.copy()\ndf_feat['energy_per_m2'] = df_feat['energy_consumption_kwh'] / df_feat['house_size_m2']\ndf_feat['building_age_group'] = pd.cut(df_feat['building_age_years'], bins=[0,10,30,60,np.inf], labels=['New','Mid-age','Old','Very old'])\ndf_feat['size_temperature_interaction'] = df_feat['house_size_m2'] * df_feat['average_temperature_c']\ndf_feat['is_old_building'] = (df_feat['building_age_years'] > 40).astype(int)\n\ndisplay(df_feat[['energy_per_m2','building_age_group','size_temperature_interaction','is_old_building']].head())"))
    cells.append(nbf.v4.new_markdown_cell("## 15) Simple preprocessing plan\n- [x] missing values handled\n- [x] duplicates checked\n- [x] outliers inspected\n- [x] categorical variables encoded\n- [x] numerical variables scaled\n- [x] engineered features created carefully"))
    cells.append(nbf.v4.new_markdown_cell("## 16) Student TODO task\n1. Create one additional engineered feature.\n2. Explain why it may help.\n3. Check missing or extreme values in that feature."))
    cells.append(nbf.v4.new_code_cell("print('Week 2 Day 1 dataset readiness practical completed successfully.')"))
    write_notebook(NB_DIR / "Week2_Day1_Data_Readiness_Feature_Engineering.ipynb", cells)


def build_day2():
    cells = []
    cells.append(nbf.v4.new_markdown_cell("# Week 2 – Day 2: Feature Selection, Data Splitting & Supervised ML Setup\n\n## 1) Introduction\nPrepared dataset -> X and y -> split -> preprocessing -> feature selection -> simple model check"))
    cells.append(nbf.v4.new_markdown_cell("## 2) Import libraries"))
    cells.append(nbf.v4.new_code_cell("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom sklearn.model_selection import train_test_split, KFold, cross_val_score\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\n\nfrom sklearn.linear_model import LinearRegression, Ridge, Lasso\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SequentialFeatureSelector\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\n%matplotlib inline"))
    cells.append(nbf.v4.new_markdown_cell("`train_test_split` splits data, `KFold/cross_val_score` run CV, `SelectKBest/RFE/SFS` perform feature selection, and `Lasso/Ridge` are embedded regularized methods."))
    cells.append(nbf.v4.new_markdown_cell(COMMON_HELPER_MD))
    cells.append(nbf.v4.new_code_cell(COMMON_HELPER_CODE))
    cells.append(nbf.v4.new_markdown_cell("## 3) Load dataset"))
    cells.append(nbf.v4.new_code_cell("if not DATASET_PATH.exists():\n    _ = create_house_energy_dataset(DATASET_PATH, n_rows=150, seed=42)\n\ndf = pd.read_csv(DATASET_PATH)\ndisplay(df.head())\nprint(df.shape)\ndf.info()\ndisplay(df.describe())"))
    cells.append(nbf.v4.new_markdown_cell("## 4) Define supervised task\nTarget is continuous (`energy_consumption_kwh`), so this is regression in a supervised setup."))
    cells.append(nbf.v4.new_code_cell("target = 'energy_consumption_kwh'\nX = df.drop(columns=[target])\ny = df[target]"))
    cells.append(nbf.v4.new_markdown_cell("## 5) Feature groups"))
    cells.append(nbf.v4.new_code_cell("numerical_features = ['house_size_m2','bedrooms','building_age_years','average_temperature_c']\ncategorical_features = ['city','heating_type','insulation_grade']\nprint(numerical_features)\nprint(categorical_features)"))
    cells.append(nbf.v4.new_markdown_cell("## 6) Train-test split"))
    cells.append(nbf.v4.new_code_cell("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nprint(X_train.shape, X_test.shape)"))
    cells.append(nbf.v4.new_markdown_cell("## 7) Data leakage warning\nDo not fit scaling/imputation/feature-selection on full data before split."))
    cells.append(nbf.v4.new_markdown_cell("## 8) Preprocessing pipeline"))
    cells.append(nbf.v4.new_code_cell("numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])\ncategorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])\npreprocessor = ColumnTransformer([('num', numerical_pipeline, numerical_features), ('cat', categorical_pipeline, categorical_features)])"))
    cells.append(nbf.v4.new_markdown_cell("## 9) Baseline model"))
    cells.append(nbf.v4.new_code_cell("baseline = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])\nbaseline.fit(X_train, y_train)\ny_pred = baseline.predict(X_test)\nmae = mean_absolute_error(y_test, y_pred)\nrmse = np.sqrt(mean_squared_error(y_test, y_pred))\nr2 = r2_score(y_test, y_pred)\nprint(mae, rmse, r2)"))
    cells.append(nbf.v4.new_markdown_cell("## 10) Correlation filter"))
    cells.append(nbf.v4.new_code_cell("corr_df = df[numerical_features + [target]].copy().fillna(df[numerical_features + [target]].median(numeric_only=True))\ncorr = corr_df.corr()[target].drop(target)\nabs_corr = corr.abs().sort_values(ascending=False)\ndisplay(pd.DataFrame({'corr': corr, 'abs_corr': corr.abs()}).sort_values('abs_corr', ascending=False))\nplt.figure(figsize=(8,4)); plt.bar(abs_corr.index, abs_corr.values); plt.title('Absolute Correlation'); plt.tight_layout(); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## 11) SelectKBest + f_regression"))
    cells.append(nbf.v4.new_code_cell("X_train_processed = preprocessor.fit_transform(X_train)\n\ndef get_feature_names(prep):\n    names = []\n    for name, transformer, cols in prep.transformers_:\n        if hasattr(transformer, 'named_steps'):\n            last = list(transformer.named_steps.values())[-1]\n            if hasattr(last, 'get_feature_names_out'):\n                names.extend(last.get_feature_names_out(cols))\n            else:\n                names.extend(cols)\n    return names\n\nfeature_names = get_feature_names(preprocessor)\nskb = SelectKBest(score_func=f_regression, k=3)\nskb.fit(X_train_processed, y_train)\nskb_df = pd.DataFrame({'feature': feature_names, 'score': skb.scores_}).sort_values('score', ascending=False)\ndisplay(skb_df.head(15))\nselected_skb = skb_df.head(3)['feature'].tolist()"))
    cells.append(nbf.v4.new_markdown_cell("## 12) Mutual Information"))
    cells.append(nbf.v4.new_code_cell("mi = mutual_info_regression(X_train_processed, y_train, random_state=42)\nmi_df = pd.DataFrame({'feature': feature_names, 'mi': mi}).sort_values('mi', ascending=False)\ndisplay(mi_df.head(15))\nplt.figure(figsize=(10,4)); plt.bar(mi_df.head(10)['feature'], mi_df.head(10)['mi']); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()\nselected_mi = mi_df.head(3)['feature'].tolist()"))
    cells.append(nbf.v4.new_markdown_cell("## 13) RFE"))
    cells.append(nbf.v4.new_code_cell("rfe = RFE(estimator=LinearRegression(), n_features_to_select=4)\nrfe.fit(X_train_processed, y_train)\nrfe_selected = [n for n, k in zip(feature_names, rfe.support_) if k]\nprint(rfe_selected)"))
    cells.append(nbf.v4.new_markdown_cell("## 14) Forward Feature Selection"))
    cells.append(nbf.v4.new_code_cell("sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=4, direction='forward', scoring='neg_mean_squared_error', cv=5)\nsfs.fit(X_train_processed, y_train)\nforward_selected = [n for n, k in zip(feature_names, sfs.get_support()) if k]\nprint(forward_selected)"))
    cells.append(nbf.v4.new_markdown_cell("## 15) Lasso"))
    cells.append(nbf.v4.new_code_cell("lasso = Lasso(alpha=0.1, max_iter=10000)\nlasso.fit(X_train_processed, y_train)\nlasso_df = pd.DataFrame({'feature': feature_names, 'lasso_coef': lasso.coef_}).sort_values('lasso_coef', key=np.abs, ascending=False)\ndisplay(lasso_df.head(15))"))
    cells.append(nbf.v4.new_markdown_cell("## 16) Ridge"))
    cells.append(nbf.v4.new_code_cell("ridge = Ridge(alpha=1.0)\nridge.fit(X_train_processed, y_train)\nridge_df = pd.DataFrame({'feature': feature_names, 'ridge_coef': ridge.coef_}).sort_values('ridge_coef', key=np.abs, ascending=False)\ndisplay(ridge_df.head(15))"))
    cells.append(nbf.v4.new_markdown_cell("## 17) Optional tree-based feature importance"))
    cells.append(nbf.v4.new_code_cell("rf = RandomForestRegressor(random_state=42, n_estimators=200)\nrf.fit(X_train_processed, y_train)\nrf_df = pd.DataFrame({'feature': feature_names, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)\ndisplay(rf_df.head(15))\nplt.figure(figsize=(10,4)); plt.bar(rf_df.head(10)['feature'], rf_df.head(10)['importance']); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## 18) Cross-validation"))
    cells.append(nbf.v4.new_code_cell("cv_model = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])\nkf = KFold(n_splits=5, shuffle=True, random_state=42)\ntry:\n    scores = cross_val_score(cv_model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')\n    rmse_scores = -scores\nexcept ValueError:\n    scores = cross_val_score(cv_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')\n    rmse_scores = np.sqrt(-scores)\nprint(np.round(rmse_scores, 2), round(rmse_scores.mean(), 2), round(rmse_scores.std(), 2))"))
    cells.append(nbf.v4.new_markdown_cell("## 19) Summary table"))
    cells.append(nbf.v4.new_code_cell("summary = pd.DataFrame([\n    ['Correlation', 'Linear relation', 'Correlations', 'Quick numeric screening'],\n    ['Mutual Information', 'Dependency incl. non-linear', 'MI scores', 'Potential non-linear effects'],\n    ['RFE', 'Remove weakest iteratively', 'Subset + ranking', 'Model-based subset search'],\n    ['Forward Selection', 'Add strongest step-by-step', 'Selected subset', 'Greedy subset building'],\n    ['Lasso', 'L1 regularization', 'Sparse coefficients', 'Feature shrinking'],\n    ['Ridge', 'L2 regularization', 'Shrunk coefficients', 'Stability'],\n    ['Random Forest Importance', 'Tree importance', 'Importance scores', 'Non-linear relationships']\n], columns=['Method', 'Main idea', 'Output', 'When useful'])\ndisplay(summary)"))
    cells.append(nbf.v4.new_markdown_cell("## 20) Student TODO task\nChoose two methods, select 3-5 features, train a simple model, compare to baseline, and write 3-5 sentence reflection."))
    cells.append(nbf.v4.new_code_cell("print('Week 2 Day 2 feature selection and splitting practical completed successfully.')"))
    write_notebook(NB_DIR / "Week2_Day2_Feature_Selection_and_Data_Splitting.ipynb", cells)


def build_day3():
    cells = []
    cells.append(nbf.v4.new_markdown_cell("# Week 2 – Day 3: Full Regression Workflow, Evaluation Metrics & Residual Analysis\n\n## 1) Introduction\nDataset -> X/y -> split -> preprocessing -> train models -> evaluate -> residual analysis -> overfitting reflection"))
    cells.append(nbf.v4.new_markdown_cell("## 2) Import libraries"))
    cells.append(nbf.v4.new_code_cell("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nfrom sklearn.model_selection import train_test_split, KFold, cross_val_score\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures\nfrom sklearn.compose import ColumnTransformer\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.linear_model import LinearRegression, Ridge, Lasso\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\n%matplotlib inline"))
    cells.append(nbf.v4.new_markdown_cell("These tools support preprocessing, model training, metric calculation, and residual visualization."))
    cells.append(nbf.v4.new_markdown_cell(COMMON_HELPER_MD))
    cells.append(nbf.v4.new_code_cell(COMMON_HELPER_CODE))
    cells.append(nbf.v4.new_markdown_cell("## 3) Load dataset"))
    cells.append(nbf.v4.new_code_cell("if not DATASET_PATH.exists():\n    _ = create_house_energy_dataset(DATASET_PATH, n_rows=150, seed=42)\n\ndf = pd.read_csv(DATASET_PATH)\ndisplay(df.head())\nprint(df.shape)\ndf.info()\ndisplay(df.describe())"))
    cells.append(nbf.v4.new_markdown_cell("## 4) Define regression task"))
    cells.append(nbf.v4.new_code_cell("target = 'energy_consumption_kwh'\nX = df.drop(columns=[target])\ny = df[target]"))
    cells.append(nbf.v4.new_markdown_cell("## 5) Train-test split"))
    cells.append(nbf.v4.new_code_cell("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\nprint(X_train.shape, X_test.shape)"))
    cells.append(nbf.v4.new_markdown_cell("## 6) Preprocessing pipeline"))
    cells.append(nbf.v4.new_code_cell("numerical_features = ['house_size_m2','bedrooms','building_age_years','average_temperature_c']\ncategorical_features = ['city','heating_type','insulation_grade']\nnum_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])\ncat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])\npreprocessor = ColumnTransformer([('num', num_pipe, numerical_features), ('cat', cat_pipe, categorical_features)])"))
    cells.append(nbf.v4.new_markdown_cell("## 7) Evaluation helper function"))
    cells.append(nbf.v4.new_code_cell("def evaluate_model(model_name, model, X_train, X_test, y_train, y_test):\n    model.fit(X_train, y_train)\n    y_train_pred = model.predict(X_train)\n    y_test_pred = model.predict(X_test)\n\n    train_mae = mean_absolute_error(y_train, y_train_pred)\n    test_mae = mean_absolute_error(y_test, y_test_pred)\n    train_mse = mean_squared_error(y_train, y_train_pred)\n    test_mse = mean_squared_error(y_test, y_test_pred)\n    train_rmse = np.sqrt(train_mse)\n    test_rmse = np.sqrt(test_mse)\n    train_r2 = r2_score(y_train, y_train_pred)\n    test_r2 = r2_score(y_test, y_test_pred)\n\n    return {\n        'Model': model_name,\n        'Train MAE': train_mae,\n        'Test MAE': test_mae,\n        'Train MSE': train_mse,\n        'Test MSE': test_mse,\n        'Train RMSE': train_rmse,\n        'Test RMSE': test_rmse,\n        'Train R2': train_r2,\n        'Test R2': test_r2,\n        'y_test_pred': y_test_pred,\n    }"))
    cells.append(nbf.v4.new_markdown_cell("MAE: average absolute error. MSE: squared error penalty. RMSE: error in original units. R2: explained variance."))
    cells.append(nbf.v4.new_markdown_cell("## 8) Model 1: Simple Linear Regression"))
    cells.append(nbf.v4.new_code_cell("X_simple = X[['average_temperature_c']]\nX_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)\nsimple_model = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('model', LinearRegression())])\nsimple_results = evaluate_model('Simple Linear Regression', simple_model, X_train_s, X_test_s, y_train_s, y_test_s)\n\nsimple_model.fit(X_train_s, y_train_s)\nplot_df = X_test_s.copy()\nplot_df['pred'] = simple_model.predict(X_test_s)\nplot_df = plot_df.sort_values('average_temperature_c')\nplt.figure(figsize=(8,5))\nplt.scatter(X_test_s['average_temperature_c'], y_test_s, alpha=0.7)\nplt.plot(plot_df['average_temperature_c'], plot_df['pred'], color='red')\nplt.title('Simple Linear Regression')\nplt.xlabel('average_temperature_c'); plt.ylabel('energy_consumption_kwh')\nplt.tight_layout(); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## 9) Model 2: Multiple Linear Regression"))
    cells.append(nbf.v4.new_code_cell("multiple_model = Pipeline([('preprocessor', preprocessor), ('model', LinearRegression())])\nmultiple_results = evaluate_model('Multiple Linear Regression', multiple_model, X_train, X_test, y_train, y_test)"))
    cells.append(nbf.v4.new_markdown_cell("## 10) Model 3: Polynomial Regression"))
    cells.append(nbf.v4.new_code_cell("poly_cols = ['average_temperature_c', 'house_size_m2', 'building_age_years']\nX_poly = X[poly_cols]\nX_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_poly, y, test_size=0.2, random_state=42)\npoly_model = Pipeline([('imputer', SimpleImputer(strategy='median')), ('poly', PolynomialFeatures(degree=2, include_bias=False)), ('scaler', StandardScaler()), ('model', LinearRegression())])\npoly_results = evaluate_model('Polynomial Regression', poly_model, X_train_p, X_test_p, y_train_p, y_test_p)"))
    cells.append(nbf.v4.new_markdown_cell("## 11) Model 4: Ridge Regression"))
    cells.append(nbf.v4.new_code_cell("ridge_model = Pipeline([('preprocessor', preprocessor), ('model', Ridge(alpha=1.0))])\nridge_results = evaluate_model('Ridge Regression', ridge_model, X_train, X_test, y_train, y_test)"))
    cells.append(nbf.v4.new_markdown_cell("## 12) Model 5: Lasso Regression"))
    cells.append(nbf.v4.new_code_cell("lasso_model = Pipeline([('preprocessor', preprocessor), ('model', Lasso(alpha=0.1, max_iter=10000))])\nlasso_results = evaluate_model('Lasso Regression', lasso_model, X_train, X_test, y_train, y_test)"))
    cells.append(nbf.v4.new_markdown_cell("## 13) Model comparison table"))
    cells.append(nbf.v4.new_code_cell("results = pd.DataFrame([{k: v for k, v in r.items() if k != 'y_test_pred'} for r in [simple_results, multiple_results, poly_results, ridge_results, lasso_results]])\nresults = results[['Model','Train MAE','Test MAE','Train RMSE','Test RMSE','Train R2','Test R2']].sort_values('Test RMSE').reset_index(drop=True)\ndisplay(results.round(3))"))
    cells.append(nbf.v4.new_markdown_cell("## 14) Visual model comparison"))
    cells.append(nbf.v4.new_code_cell("plt.figure(figsize=(10,4)); plt.bar(results['Model'], results['Test RMSE']); plt.xticks(rotation=25, ha='right'); plt.title('Test RMSE'); plt.tight_layout(); plt.show()\nplt.figure(figsize=(10,4)); plt.bar(results['Model'], results['Test R2']); plt.xticks(rotation=25, ha='right'); plt.title('Test R2'); plt.tight_layout(); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## 15) Actual vs predicted plot"))
    cells.append(nbf.v4.new_code_cell("best_model_name = results.iloc[0]['Model']\nresult_map = {'Simple Linear Regression': simple_results, 'Multiple Linear Regression': multiple_results, 'Polynomial Regression': poly_results, 'Ridge Regression': ridge_results, 'Lasso Regression': lasso_results}\nbest_result = result_map[best_model_name]\ny_pred_best = best_result['y_test_pred']\nif best_model_name == 'Simple Linear Regression':\n    y_test_best = y_test_s\nelif best_model_name == 'Polynomial Regression':\n    y_test_best = y_test_p\nelse:\n    y_test_best = y_test\n\nplt.figure(figsize=(6,6))\nplt.scatter(y_test_best, y_pred_best, alpha=0.7)\nmn, mx = min(y_test_best.min(), y_pred_best.min()), max(y_test_best.max(), y_pred_best.max())\nplt.plot([mn, mx], [mn, mx], 'r--')\nplt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title(f'Actual vs Predicted ({best_model_name})')\nplt.tight_layout(); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## 16) Residual analysis"))
    cells.append(nbf.v4.new_code_cell("residuals = y_test_best - y_pred_best\nplt.figure(figsize=(8,4)); plt.scatter(y_pred_best, residuals, alpha=0.7); plt.axhline(0, color='red', linestyle='--'); plt.title('Residuals vs Predicted'); plt.tight_layout(); plt.show()\nplt.figure(figsize=(8,4)); plt.hist(residuals, bins=12, edgecolor='black'); plt.title('Residual Histogram'); plt.tight_layout(); plt.show()\nplt.figure(figsize=(8,4)); plt.scatter(y_test_best, residuals, alpha=0.7); plt.axhline(0, color='red', linestyle='--'); plt.title('Residuals vs Actual'); plt.tight_layout(); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("Residual interpretation:\n- random around zero: model is reasonable\n- curved pattern: model may be too simple\n- increasing spread: unequal variance\n- large isolated points: possible outliers\n- visible structure: missing features or wrong model form"))
    cells.append(nbf.v4.new_markdown_cell("## 17) Overfitting and underfitting reflection\nOverfitting: strong train but weak test. Underfitting: both train and test weak.\n\nQuestions:\n- Which model is best?\n- Is there overfitting?\n- Is there underfitting?\n- How could we improve?"))
    cells.append(nbf.v4.new_markdown_cell("## 18) Cross-validation final check"))
    cells.append(nbf.v4.new_code_cell("model_map = {'Simple Linear Regression': simple_model, 'Multiple Linear Regression': multiple_model, 'Polynomial Regression': poly_model, 'Ridge Regression': ridge_model, 'Lasso Regression': lasso_model}\nbest_pipeline = model_map[best_model_name]\nkf = KFold(n_splits=5, shuffle=True, random_state=42)\nif best_model_name == 'Simple Linear Regression':\n    X_cv, y_cv = X_train_s, y_train_s\nelif best_model_name == 'Polynomial Regression':\n    X_cv, y_cv = X_train_p, y_train_p\nelse:\n    X_cv, y_cv = X_train, y_train\ntry:\n    cv_scores = cross_val_score(best_pipeline, X_cv, y_cv, cv=kf, scoring='neg_root_mean_squared_error')\n    cv_rmse = -cv_scores\nexcept ValueError:\n    cv_scores = cross_val_score(best_pipeline, X_cv, y_cv, cv=kf, scoring='neg_mean_squared_error')\n    cv_rmse = np.sqrt(-cv_scores)\nprint('RMSE per fold:', np.round(cv_rmse, 2))\nprint('Mean RMSE:', round(cv_rmse.mean(), 2))\nprint('Std RMSE:', round(cv_rmse.std(), 2))"))
    cells.append(nbf.v4.new_markdown_cell("## 19) Optional note: Logistic Regression\nLogistic regression is mainly for classification, not for predicting continuous values, so it is not a main regression model here."))
    cells.append(nbf.v4.new_code_cell("x = np.linspace(-10, 10, 200)\ny = 1 / (1 + np.exp(-x))\nplt.figure(figsize=(7,4)); plt.plot(x, y); plt.title('Optional sigmoid curve'); plt.tight_layout(); plt.show()"))
    cells.append(nbf.v4.new_markdown_cell("## 20) Final student task\n1. Define X and y\n2. Split data correctly\n3. Apply preprocessing\n4. Train baseline\n5. Train improved model\n6. Evaluate MAE/MSE/RMSE/R2\n7. Compare models\n8. Plot residuals\n9. Interpret error patterns\n10. Reflect on overfitting/underfitting"))
    cells.append(nbf.v4.new_code_cell("print('Week 2 Day 3 full regression workflow practical completed successfully.')"))
    write_notebook(NB_DIR / "Week2_Day3_Full_Regression_Workflow.ipynb", cells)


def build_readme():
    readme = """# Week 2 Practical Notebooks

This folder contains the three live-coding practical notebooks for **Week 2: Machine Learning Foundations & Regression Workflow**.

## Day 1
- Topic: Dataset readiness, preprocessing, feature engineering
- Notebook: `Week2_Day1_Data_Readiness_Feature_Engineering.ipynb`

## Day 2
- Topic: Feature selection, data splitting, cross-validation
- Notebook: `Week2_Day2_Feature_Selection_and_Data_Splitting.ipynb`

## Day 3
- Topic: Full regression workflow, evaluation metrics, residual analysis
- Notebook: `Week2_Day3_Full_Regression_Workflow.ipynb`

## Shared dataset
- Path: `data/week2/house_energy_week2_sample.csv`

## Learning goals
- Build a week-long connected workflow from raw data readiness to regression evaluation.
- Apply leakage-safe preprocessing and train/test split.
- Practice feature-selection methods and model comparison.
- Interpret model errors and residual patterns.

## Expected student outputs
- Cleaned and prepared dataset.
- Feature-selection comparison and supervised setup.
- Regression model comparison with metrics and residual interpretation.

> Run **Day 1 first** because it creates the shared dataset for Day 2 and Day 3.
"""
    (NB_DIR / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    build_day1()
    build_day2()
    build_day3()
    build_readme()
    print("Week 2 notebooks and README generated.")
