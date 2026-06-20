"""Regenerate Week 1 Lecture_Support notebooks (beginner-friendly, synthetic data)."""
from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).resolve().parents[1]
NB_DIR = BASE / "Week_01_Statistical_Foundations_and_Inference" / "Lecture_Support"
NB_DIR.mkdir(parents=True, exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def write_notebook(path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)


def header(title, objectives):
    return [
        md(f"# {title}\n"),
        md("## 1) Learning Objectives\n\n" + objectives),
        md(
            "## 2) Why This Topic Matters\n\n"
            "Statistics starts with understanding your data. "
            "If we skip inspection or mix up variable types, later analysis can become misleading."
        ),
    ]


def footer(concept_q, code_q, takeaways, practice, summary=None, refinement1="", refinement2=""):
    summary_text = summary or "Review the tables and plots above and connect each numeric result to a plain-language sentence."
    return [
        md(f"## 7) Summary Interpretation\n\n{summary_text}"),
        md(
            "## 8) Student Check Questions\n\n"
            f"1. (Conceptual) {concept_q}\n\n"
            f"2. (Code) {code_q}"
        ),
        code("# Question 2 — your code here\n"),
        md("## 9) Key Takeaways\n\n" + takeaways),
        md("## 10) Optional Short Practice / Challenge\n\n" + practice),
        md(
            "## 11) Code Refinement Tasks\n\n"
            "Edit the code, rerun each cell, and write one short sentence per task.\n\n"
            f"1. {refinement1}\n"
            f"2. {refinement2}"
        ),
        code("# Refinement 1\n"),
        code("# Refinement 2\n"),
    ]


def build_nb01():
    cells = header(
        "01 - Data Types and Descriptive Statistics",
        "- Identify qualitative and quantitative variables.\n"
        "- Inspect a dataset correctly before analysis.\n"
        "- Compute and interpret descriptive statistics.\n"
        "- Compare groups with simple summaries.\n"
        "- Connect summary tables to real analysis decisions.",
    )
    cells += [
        md(
            "## 3) Short and Simple Theory Explanation\n\n"
            "- **Qualitative (categorical) variable**: labels or categories (for example, city, gender).\n"
            "- **Quantitative (numerical) variable**: numbers that measure something (for example, age, score).\n"
            "- **Descriptive statistics** summarize data using center, spread, and counts."
        ),
        md("## 4) Step-by-Step Code Examples"),
        md(
            "### Step 4.1 — Import libraries\n\n"
            "- `import` loads Python packages we need.\n"
            "- `numpy` (`np`) helps with numbers and random data.\n"
            "- `pandas` (`pd`) helps with tables called **DataFrames**.\n"
            "- `matplotlib.pyplot` (`plt`) creates plots.\n"
            "- `np.random.seed(42)` makes random results repeatable."
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "np.random.seed(42)\n"
            "sns.set_theme(style='whitegrid')\n"
            "print('Libraries loaded successfully.')"
        ),
        md(
            "**Interpretation:** Setup is complete. Using the same seed each time helps classmates get the same synthetic dataset."
        ),
        md(
            "### Step 4.2 — Create a synthetic student dataset\n\n"
            "We build a small table inside the notebook (no external file needed).\n"
            "A **DataFrame** is like an Excel sheet in Python: rows are observations, columns are variables."
        ),
        code(
            "rng = np.random.default_rng(42)\n"
            "number_of_students = 120\n\n"
            "students = pd.DataFrame({\n"
            "    'student_id': np.arange(1, number_of_students + 1),\n"
            "    'age': rng.integers(18, 26, size=number_of_students),\n"
            "    'exam_score': np.round(rng.normal(72, 10, number_of_students), 1),\n"
            "    'study_hours': np.round(rng.normal(5, 2, number_of_students).clip(0.5, 14), 1),\n"
            "    'part_time_income': np.round(rng.normal(450, 120, number_of_students).clip(0, None), 0),\n"
            "    'gender': rng.choice(['Female', 'Male', 'Other'], size=number_of_students, p=[0.48, 0.48, 0.04]),\n"
            "    'study_group': rng.choice(['Morning', 'Evening', 'Weekend'], size=number_of_students),\n"
            "    'city': rng.choice(['Berlin', 'Munich', 'Hamburg'], size=number_of_students),\n"
            "})\n\n"
            "print('Dataset shape (rows, columns):', students.shape)\n"
            "students.head()"
        ),
        md(
            "**What is `head()`?** It shows the first 5 rows so we can quickly see column names and example values."
        ),
        md(
            "### Step 4.3 — Inspect structure with `info()` and `describe()`\n\n"
            "- `info()` shows column names, data types, and missing values.\n"
            "- `describe()` computes basic numerical summaries (mean, std, min, max, quartiles)."
        ),
        code(
            "print('--- info() ---')\n"
            "students.info()\n"
            "print('\\n--- describe() for numerical columns ---')\n"
            "students.describe().round(2)"
        ),
        md(
            "**Interpretation:** `describe()` gives a fast overview of numerical columns. "
            "For example, average exam score and how much scores vary (std)."
        ),
        md(
            "### Step 4.4 — Identify variable types explicitly\n\n"
            "We separate columns into **quantitative** (numbers) and **qualitative** (categories)."
        ),
        code(
            "quantitative_columns = students.select_dtypes(include='number').columns.tolist()\n"
            "qualitative_columns = students.select_dtypes(exclude='number').columns.tolist()\n\n"
            "print('Quantitative columns:', quantitative_columns)\n"
            "print('Qualitative columns:', qualitative_columns)\n\n"
            "print('\\nValue counts for study_group:')\n"
            "print(students['study_group'].value_counts())"
        ),
        md(
            "**What is `value_counts()`?** It counts how often each category appears. "
            "This is useful for qualitative variables."
        ),
        md(
            "### Step 4.5 — Grouped descriptive statistics\n\n"
            "`groupby()` splits the table into groups, then we apply summary functions such as `mean()`."
        ),
        code(
            "grouped_means = students.groupby('study_group')[['exam_score', 'study_hours']].mean().round(2)\n"
            "grouped_means"
        ),
        md(
            "**Interpretation:** Morning, Evening, and Weekend groups can have different average scores. "
            "Grouped summaries reveal patterns that one overall mean may hide."
        ),
        code(
            "grouped_full = students.groupby('study_group').agg({\n"
            "    'exam_score': ['mean', 'median', 'std'],\n"
            "    'study_hours': ['mean', 'median'],\n"
            "}).round(2)\n"
            "grouped_full"
        ),
        md(
            "### Step 4.6 — Simple bar plot of group means\n\n"
            "`plt.bar()` draws a bar chart. We plot average exam score by study group."
        ),
        code(
            "mean_scores = students.groupby('study_group')['exam_score'].mean()\n\n"
            "plt.figure(figsize=(7, 4))\n"
            "plt.bar(mean_scores.index, mean_scores.values, color='steelblue')\n"
            "plt.title('Average exam score by study group')\n"
            "plt.xlabel('Study group')\n"
            "plt.ylabel('Mean exam score')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "**Interpretation:** The plot makes group differences easier to see than numbers alone."
        ),
        md(
            "### Step 4.7 — Manual mean example (build intuition)\n\n"
            "Packages compute quickly, but a manual example shows what `mean()` does."
        ),
        code(
            "small_scores = students['exam_score'].head(5)\n"
            "print('First 5 exam scores:')\n"
            "print(small_scores.values)\n\n"
            "manual_mean = small_scores.sum() / len(small_scores)\n"
            "pandas_mean = small_scores.mean()\n\n"
            "print('Manual mean:', round(manual_mean, 2))\n"
            "print('pandas mean():', round(pandas_mean, 2))"
        ),
        md("**Interpretation:** Both methods match. Mean = sum of values divided by count."),
    ]
    cells += footer(
        "Why separate qualitative and quantitative variables?",
        "Group by `city` instead of `study_group` and compare mean `exam_score`.",
        "- Inspect before interpreting.\n"
        "- Variable types are foundational.\n"
        "- Grouped descriptive statistics are more informative than one global number.\n"
        "- Manual checks improve statistical intuition.",
        "Convert `gender` and `city` to category dtype, then run `info()`.",
        summary="We created and inspected a student table, separated variable types, and compared groups with tables and a bar chart.",
        refinement1="Group by `city` and show mean `exam_score`.",
        refinement2="Keep only students with `age >= 20` and run `describe()` on `exam_score`.",
    )
    cells.append(code("print('Notebook 01 completed successfully.')"))
    write_notebook(NB_DIR / "01_Data_Types_and_Descriptive_Statistics.ipynb", cells)


def build_nb02():
    cells = header(
        "02 - Central Tendency, Dispersion, and Visualization",
        "- Compute mean, median, and mode clearly.\n"
        "- Compare variance, standard deviation, and range.\n"
        "- Explain when median is better than mean.\n"
        "- Use basic plots to support interpretation.",
    )
    cells += [
        md(
            "## 3) Short and Simple Theory Explanation\n\n"
            "- **Mean**: average; sensitive to outliers.\n"
            "- **Median**: middle value; more robust to outliers.\n"
            "- **Mode**: most frequent value.\n"
            "- **Variance** (`var()`): average squared distance from mean.\n"
            "- **Standard deviation** (`std()`): square root of variance, in original units.\n"
            "- **Range**: maximum minus minimum."
        ),
        md("## 4) Step-by-Step Code Examples"),
        md("### Step 4.1 — Setup and create synthetic data"),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "np.random.seed(42)\n"
            "sns.set_theme(style='whitegrid')\n\n"
            "rng = np.random.default_rng(42)\n"
            "number_of_rows = 100\n\n"
            "normal_scores = rng.normal(loc=70, scale=8, size=number_of_rows)\n"
            "scores_with_outlier = normal_scores.copy()\n"
            "scores_with_outlier[0] = 150  # one extreme outlier\n\n"
            "course_data = pd.DataFrame({\n"
            "    'normal_score': np.round(normal_scores, 1),\n"
            "    'score_with_outlier': np.round(scores_with_outlier, 1),\n"
            "    'study_group': rng.choice(['Group A', 'Group B'], size=number_of_rows),\n"
            "})\n\n"
            "course_data.head()"
        ),
        md("### Step 4.2 — Mean, median, and mode"),
        code(
            "column_name = 'normal_score'\n\n"
            "mean_value = course_data[column_name].mean()\n"
            "median_value = course_data[column_name].median()\n"
            "mode_series = course_data[column_name].mode()\n\n"
            "print('Mean:', round(mean_value, 2))\n"
            "print('Median:', round(median_value, 2))\n"
            "print('Mode (first mode if several):', mode_series.iloc[0])"
        ),
        md(
            "**Interpretation:** For the normal-looking variable, mean and median are close. "
            "That often happens when data are symmetric."
        ),
        md("### Step 4.3 — Effect of one outlier on mean vs median"),
        code(
            "for col in ['normal_score', 'score_with_outlier']:\n"
            "    print('\\nColumn:', col)\n"
            "    print('  Mean  :', round(course_data[col].mean(), 2))\n"
            "    print('  Median:', round(course_data[col].median(), 2))"
        ),
        md(
            "**Interpretation:** After adding one very high value, the mean jumps up but the median changes much less. "
            "That is why median is often preferred when outliers exist."
        ),
        md("### Step 4.4 — Dispersion: variance, standard deviation, range"),
        code(
            "col = 'normal_score'\n"
            "variance_value = course_data[col].var()\n"
            "std_value = course_data[col].std()\n"
            "range_value = course_data[col].max() - course_data[col].min()\n\n"
            "summary = pd.DataFrame({\n"
            "    'variance': [variance_value],\n"
            "    'std': [std_value],\n"
            "    'range': [range_value],\n"
            "}).round(2)\n"
            "summary"
        ),
        md(
            "**Interpretation:** Standard deviation is easier to interpret than variance because it uses the same units as the data (points)."
        ),
        md("### Step 4.5 — Group comparison"),
        code(
            "group_summary = course_data.groupby('study_group')['normal_score'].agg(['mean', 'median', 'std']).round(2)\n"
            "group_summary"
        ),
        md("### Step 4.6 — Visualize with histogram, boxplot, and scatter plot"),
        code(
            "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n\n"
            "axes[0].hist(course_data['normal_score'], bins=15, color='steelblue', edgecolor='white')\n"
            "axes[0].set_title('Histogram')\n"
            "axes[0].set_xlabel('Score')\n\n"
            "course_data.boxplot(column='score_with_outlier', by='study_group', ax=axes[1])\n"
            "axes[1].set_title('Boxplot (with outlier)')\n"
            "axes[1].set_xlabel('Study group')\n\n"
            "axes[2].scatter(course_data['normal_score'], course_data['score_with_outlier'], alpha=0.7)\n"
            "axes[2].set_title('Scatter plot')\n"
            "axes[2].set_xlabel('normal_score')\n"
            "axes[2].set_ylabel('score_with_outlier')\n\n"
            "plt.suptitle('')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "**Interpretation:**\n"
            "- The **histogram** shows how values are spread.\n"
            "- The **boxplot** highlights the outlier as a point far from the box.\n"
            "- The **scatter plot** shows relationship between two numeric columns."
        ),
        md("### Step 4.7 — Add or remove an outlier and observe changes"),
        code(
            "without_outlier = course_data['score_with_outlier'].copy()\n"
            "without_outlier.iloc[0] = 72  # replace extreme value with a typical score\n\n"
            "comparison = pd.DataFrame({\n"
            "    'with_outlier_mean': [course_data['score_with_outlier'].mean()],\n"
            "    'with_outlier_median': [course_data['score_with_outlier'].median()],\n"
            "    'without_outlier_mean': [without_outlier.mean()],\n"
            "    'without_outlier_median': [without_outlier.median()],\n"
            "}).round(2)\n"
            "comparison"
        ),
        md("**Interpretation:** Removing the outlier brings the mean closer to the median."),
    ]
    cells += footer(
        "When is median better than mean?",
        "Add outlier score 200 and compare mean vs median.",
        "- Report center and spread together.\n"
        "- Use plots to support tables.\n"
        "- Prefer median when extreme values are present.\n"
        "- Always interpret numbers in context.",
        "Add a skewed column with `rng.exponential(scale=10, size=number_of_rows)`.",
        summary="We compared mean, median, and spread measures and saw how one outlier can distort the mean but not the median.",
        refinement1="Add outliers 180 and 200 to `score_with_outlier`; print mean and median.",
        refinement2="Change histogram bins from 15 to 5 and replot.",
    )
    cells.append(code("print('Notebook 02 completed successfully.')"))
    write_notebook(NB_DIR / "02_Central_Tendency_Dispersion_and_Visualization.ipynb", cells)


def build_nb03():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from week1_nb03_05 import build_nb03 as _cells
    write_notebook(NB_DIR / "03_Distributions_Skewness_Sampling_and_CLT.ipynb", _cells())


def build_nb04():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from week1_nb03_05 import build_nb04 as _cells
    write_notebook(NB_DIR / "04_Inferential_Statistics_ttest_ANOVA_Correlation.ipynb", _cells())


def build_nb05():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from week1_nb03_05 import build_nb05 as _cells
    write_notebook(NB_DIR / "05_One_Question_One_Test_One_Interpretation.ipynb", _cells())


def update_readme():
    text = """# Week 1: Statistical Foundations and Inference

This week builds core statistical thinking for later machine learning topics.

## Lecture_Support notebooks (run in order)

1. `01_Data_Types_and_Descriptive_Statistics.ipynb`
2. `02_Central_Tendency_Dispersion_and_Visualization.ipynb`
3. `03_Distributions_Skewness_Sampling_and_CLT.ipynb`
4. `04_Inferential_Statistics_ttest_ANOVA_Correlation.ipynb`
5. `05_One_Question_One_Test_One_Interpretation.ipynb`

All notebooks use **synthetic datasets created inside the notebook** (no external files required).

## Folder purpose

- `Lecture_Support/` — guided lecture examples with interpretation
- `Practical_Lab/` — in-class tasks with hints and solutions

## Regenerate notebooks

```bash
python3 scripts/build_week1_notebooks.py
```
"""
    readme = BASE / "Week_01_Statistical_Foundations_and_Inference" / "README.md"
    readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    build_nb01()
    build_nb02()
    build_nb03()
    build_nb04()
    build_nb05()
    update_readme()
    print("Week 1 notebooks 01-05 generated in Lecture_Support.")
