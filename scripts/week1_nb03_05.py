"""Week 1 notebooks 03-05 cell builders."""
from build_week1_notebooks import md, code, header, footer


def build_nb03():
    cells = header(
        "03 - Distributions, Skewness, Sampling, and CLT",
        "- Generate normal, uniform, Poisson, and skewed synthetic data.\n"
        "- Explain distribution shape differences.\n"
        "- Compute and interpret skewness.\n"
        "- Demonstrate sampling variability and the Central Limit Theorem (CLT).",
    )
    cells += [
        md(
            "## 3) Short and Simple Theory Explanation\n\n"
            "- A **distribution** describes how values are spread.\n"
            "- **Skewness** measures asymmetry (long tail on one side).\n"
            "- **Sampling** means drawing random subsets from a population.\n"
            "- **CLT:** sample means become more bell-shaped as sample size increases, even if the original population is skewed."
        ),
        md("## 4) Step-by-Step Code Examples"),
        md(
            "### Step 4.1 — Setup\n\n"
            "We use `numpy` to generate random numbers and `scipy.stats.skew` to measure skewness."
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from scipy.stats import skew\n\n"
            "np.random.seed(42)\n"
            "sns.set_theme(style='whitegrid')\n"
            "rng = np.random.default_rng(42)"
        ),
        md("### Step 4.2 — Generate four synthetic distributions"),
        code(
            "sample_size = 1000\n\n"
            "normal_data = rng.normal(loc=50, scale=10, size=sample_size)\n"
            "uniform_data = rng.uniform(low=0, high=100, size=sample_size)\n"
            "poisson_data = rng.poisson(lam=8, size=sample_size)\n"
            "skewed_data = rng.exponential(scale=15, size=sample_size)\n\n"
            "distribution_table = pd.DataFrame({\n"
            "    'distribution': ['Normal', 'Uniform', 'Poisson', 'Skewed (Exponential)'],\n"
            "    'mean': [normal_data.mean(), uniform_data.mean(), poisson_data.mean(), skewed_data.mean()],\n"
            "    'variance': [normal_data.var(), uniform_data.var(), poisson_data.var(), skewed_data.var()],\n"
            "    'skewness': [skew(normal_data), skew(uniform_data), skew(poisson_data), skew(skewed_data)],\n"
            "}).round(3)\n"
            "distribution_table"
        ),
        md(
            "**Interpretation:** Different generators create different shapes. "
            "Poisson data are counts (non-negative integers). Exponential data are right-skewed."
        ),
        md("### Step 4.3 — Visualize each distribution with histograms"),
        code(
            "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\n"
            "datasets = [\n"
            "    (normal_data, 'Normal'),\n"
            "    (uniform_data, 'Uniform'),\n"
            "    (poisson_data, 'Poisson'),\n"
            "    (skewed_data, 'Skewed (Exponential)'),\n"
            "]\n"
            "for ax, (values, title) in zip(axes.ravel(), datasets):\n"
            "    ax.hist(values, bins=30, color='steelblue', edgecolor='white')\n"
            "    ax.set_title(title)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("**Interpretation:** Normal looks bell-shaped; uniform looks flat; skewed has a long right tail."),
        md("### Step 4.4 — Skewness values in plain language"),
        code(
            "for name, values in [('Normal', normal_data), ('Skewed', skewed_data)]:\n"
            "    print(f\"{name}: skewness = {skew(values):.3f}\")"
        ),
        md(
            "**Interpretation:** Skewness near 0 suggests symmetry. Positive skewness means a longer right tail."
        ),
        md("### Step 4.5 — Build a skewed population for CLT demo"),
        code(
            "population = rng.exponential(scale=20, size=5000)\n"
            "plt.figure(figsize=(7, 4))\n"
            "plt.hist(population, bins=40, color='darkorange', edgecolor='white')\n"
            "plt.title('Original population (skewed, not normal)')\n"
            "plt.xlabel('Value')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print('Population mean:', round(population.mean(), 2))"
        ),
        md("### Step 4.6 — Sampling experiment and CLT"),
        code(
            "def sample_means(data, sample_size, number_of_samples):\n"
            "    means = []\n"
            "    for _ in range(number_of_samples):\n"
            "        one_sample = rng.choice(data, size=sample_size, replace=True)\n"
            "        means.append(one_sample.mean())\n"
            "    return np.array(means)\n\n"
            "small_sample_means = sample_means(population, sample_size=5, number_of_samples=2000)\n"
            "large_sample_means = sample_means(population, sample_size=50, number_of_samples=2000)\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
            "axes[0].hist(small_sample_means, bins=30, color='steelblue', edgecolor='white')\n"
            "axes[0].set_title('Sample means (n = 5)')\n"
            "axes[1].hist(large_sample_means, bins=30, color='seagreen', edgecolor='white')\n"
            "axes[1].set_title('Sample means (n = 50)')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "**Interpretation:** Even though the population is skewed, the distribution of sample means becomes more bell-shaped when sample size increases. "
            "That is the idea behind the CLT."
        ),
        md("### Step 4.7 — Change sample size and observe"),
        code(
            "medium_sample_means = sample_means(population, sample_size=20, number_of_samples=2000)\n"
            "clt_compare = pd.DataFrame({\n"
            "    'sample_size': [5, 20, 50],\n"
            "    'mean_of_sample_means': [small_sample_means.mean(), medium_sample_means.mean(), large_sample_means.mean()],\n"
            "    'std_of_sample_means': [small_sample_means.std(), medium_sample_means.std(), large_sample_means.std()],\n"
            "}).round(3)\n"
            "clt_compare"
        ),
        md("**Interpretation:** The mean of sample means stays near the population mean, while spread shrinks as sample size grows."),
    ]
    cells += footer(
        "What does the Central Limit Theorem tell us about sample means?",
        "It says that sample means tend toward a normal distribution as sample size increases, even when the original population is not normal.",
        "Change `sample_size` from 5 to 100 in the sampling function and replot the histogram of sample means.",
        "- Distributions describe how data are generated.\n"
        "- Skewness helps describe asymmetry.\n"
        "- Sampling introduces random variation.\n"
        "- Larger samples give more stable mean estimates.",
        "Generate a binomial distribution with `rng.binomial(n=20, p=0.3, size=1000)` and add it to the comparison table.",
        summary="We generated different distributions, measured skewness, and showed that sample means become more normal as sample size grows.",
        refinement1=(
            "In **Step 4.6**, change `sample_size=5` to `sample_size=100` when calling `sample_means(...)`. "
            "Plot the new histogram of sample means and compare it with the small-sample plot. "
            "Does it look more bell-shaped?"
        ),
        refinement2=(
            "In **Step 4.2**, change the Poisson parameter from `lam=8` to `lam=20`. "
            "Recompute mean, variance, and skewness for Poisson data and explain whether the mean increased as you expected."
        ),
    )
    cells.append(code("print('Notebook 03 completed successfully.')"))
    return cells


def build_nb04():
    cells = header(
        "04 - Inferential Statistics: t-test, ANOVA, Correlation",
        "- Explain hypothesis-testing workflow.\n"
        "- Interpret p-values correctly.\n"
        "- Compute and explain correlation.\n"
        "- Apply t-test (2 groups) and ANOVA (3 groups).",
    )
    cells += [
        md(
            "## 3) Short and Simple Theory Explanation\n\n"
            "- **Hypothesis:** a testable statement about data.\n"
            "- **H0 (null):** no difference / no relationship.\n"
            "- **H1 (alternative):** difference or relationship exists.\n"
            "- **p-value:** probability of seeing data this extreme if H0 were true.\n"
            "- **t-test:** compares means of two groups.\n"
            "- **ANOVA:** compares means across three or more groups.\n"
            "- **Correlation:** measures linear association; it does not prove causation."
        ),
        md("## 4) Step-by-Step Code Examples"),
        md("### Step 4.1 — Import tools"),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from scipy import stats\n\n"
            "np.random.seed(42)\n"
            "sns.set_theme(style='whitegrid')\n"
            "rng = np.random.default_rng(42)"
        ),
        md("### Step 4.2 — Synthetic two-group dataset for t-test"),
        code(
            "group_a_scores = rng.normal(loc=72, scale=8, size=40)\n"
            "group_b_scores = rng.normal(loc=78, scale=8, size=40)\n\n"
            "ttest_data = pd.DataFrame({\n"
            "    'score': np.concatenate([group_a_scores, group_b_scores]),\n"
            "    'method': ['Method A'] * 40 + ['Method B'] * 40,\n"
            "})\n\n"
            "ttest_data.groupby('method')['score'].agg(['mean', 'std', 'count']).round(2)"
        ),
        md(
            "**Interpretation:** Method B has a higher sample mean, but we still ask whether the difference is statistically meaningful."
        ),
        md("### Step 4.3 — Independent two-sample t-test"),
        code(
            "scores_a = ttest_data.loc[ttest_data['method'] == 'Method A', 'score']\n"
            "scores_b = ttest_data.loc[ttest_data['method'] == 'Method B', 'score']\n\n"
            "t_statistic, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)\n"
            "print('t statistic:', round(t_statistic, 3))\n"
            "print('p-value:', round(p_value, 4))\n\n"
            "if p_value < 0.05:\n"
            "    print('At alpha = 0.05, we reject H0: groups appear different on average.')\n"
            "else:\n"
            "    print('At alpha = 0.05, we do not reject H0: difference may be due to chance.')"
        ),
        md(
            "**What is `stats.ttest_ind`?** It runs an independent two-sample t-test and returns a test statistic and p-value."
        ),
        md("### Step 4.4 — Synthetic three-group dataset for ANOVA"),
        code(
            "anova_data = pd.DataFrame({\n"
            "    'score': np.concatenate([\n"
            "        rng.normal(70, 7, 35),\n"
            "        rng.normal(75, 7, 35),\n"
            "        rng.normal(82, 7, 35),\n"
            "    ]),\n"
            "    'program': ['Program 1'] * 35 + ['Program 2'] * 35 + ['Program 3'] * 35,\n"
            "})\n\n"
            "anova_data.groupby('program')['score'].mean().round(2)"
        ),
        code(
            "group1 = anova_data.loc[anova_data['program'] == 'Program 1', 'score']\n"
            "group2 = anova_data.loc[anova_data['program'] == 'Program 2', 'score']\n"
            "group3 = anova_data.loc[anova_data['program'] == 'Program 3', 'score']\n\n"
            "f_statistic, p_value_anova = stats.f_oneway(group1, group2, group3)\n"
            "print('F statistic:', round(f_statistic, 3))\n"
            "print('p-value:', round(p_value_anova, 4))"
        ),
        md("**Interpretation:** A small p-value suggests at least one group mean differs, but ANOVA does not tell us which pair differs."),
        md("### Step 4.5 — Correlation between two numerical variables"),
        code(
            "study_hours = rng.normal(5, 1.5, 60)\n"
            "exam_scores = 50 + 4 * study_hours + rng.normal(0, 3, 60)\n"
            "correlation_data = pd.DataFrame({'study_hours': study_hours, 'exam_score': exam_scores})\n\n"
            "correlation_value = correlation_data['study_hours'].corr(correlation_data['exam_score'])\n"
            "print('Pearson correlation:', round(correlation_value, 3))\n\n"
            "plt.figure(figsize=(6, 4))\n"
            "plt.scatter(correlation_data['study_hours'], correlation_data['exam_score'], alpha=0.7)\n"
            "plt.xlabel('Study hours')\n"
            "plt.ylabel('Exam score')\n"
            "plt.title('Scatter plot for correlation')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "**Interpretation:** Positive correlation means higher study hours tend to appear with higher scores. "
            "This does **not** prove that study hours alone cause higher scores."
        ),
        md("### Step 4.6 — Change noise level and observe p-value"),
        code(
            "def run_ttest(mean_diff, noise_level):\n"
            "    a = rng.normal(70, noise_level, 40)\n"
            "    b = rng.normal(70 + mean_diff, noise_level, 40)\n"
            "    _, p = stats.ttest_ind(a, b, equal_var=False)\n"
            "    return p\n\n"
            "p_demo = pd.DataFrame({\n"
            "    'mean_difference': [0, 2, 8],\n"
            "    'noise_level': [8, 8, 8],\n"
            "    'p_value': [run_ttest(0, 8), run_ttest(2, 8), run_ttest(8, 8)],\n"
            "}).round(4)\n"
            "p_demo"
        ),
        md(
            "**Interpretation:** Larger true differences and lower noise usually produce smaller p-values. "
            "A p-value is not proof of importance or causation."
        ),
    ]
    cells += footer(
        "What does a p-value tell us, and what does it not prove?",
        "A p-value measures evidence against H0 under the test assumptions. It does not prove causation, practical importance, or that H1 is true.",
        "Increase the noise level in the t-test demo (for example, scale = 20) and observe how the p-value changes.",
        "- State H0 and H1 before testing.\n"
        "- Choose the test that matches your question and number of groups.\n"
        "- Report effect size and context, not only p-values.\n"
        "- Correlation is not causation.",
        "Run `stats.pearsonr` on the correlation dataset and compare with pandas `.corr()`.",
        summary="We practiced t-tests, ANOVA, and correlation on synthetic data and interpreted p-values carefully.",
        refinement1=(
            "In **Step 4.6**, change `noise_level` from `8` to `20` in the t-test demo and rerun the table of p-values. "
            "Do the p-values become larger or smaller? Why?"
        ),
        refinement2=(
            "In **Step 4.3**, reduce the mean difference between Method A and Method B "
            "(for example, use `loc=72` and `loc=73` instead of `72` and `78`). "
            "Rerun the t-test and check whether the p-value increases."
        ),
    )
    cells.append(code("print('Notebook 04 completed successfully.')"))
    return cells


def build_nb05():
    cells = [
        md("# 05 - One Question, One Test, One Interpretation\n"),
        md(
            "## 1) Learning Objectives\n\n"
            "- Connect population vs sample reasoning.\n"
            "- Perform short exploratory data analysis (EDA).\n"
            "- State H0 and H1 clearly.\n"
            "- Run exactly one inferential test.\n"
            "- Interpret p-value and confidence interval in plain language."
        ),
        md(
            "## 2) Why This Topic Matters\n\n"
            "Good analysis starts with **one clear question**. "
            "Running many tests without planning increases the risk of false discoveries."
        ),
        md(
            "## 3) Our One Question\n\n"
            "**Question:** Do students who use **Method A** and **Method B** have different average exam scores?\n\n"
            "This fits Week 1 because we have:\n"
            "- two groups,\n"
            "- one numeric outcome,\n"
            "- one test: independent two-sample t-test."
        ),
        md("## 4) Step-by-Step Workflow"),
        md(
            "### Step 4.1 — Setup and create synthetic data\n\n"
            "We simulate a sample of students from two teaching methods."
        ),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from scipy import stats\n\n"
            "np.random.seed(42)\n"
            "sns.set_theme(style='whitegrid')\n"
            "rng = np.random.default_rng(42)\n\n"
            "sample_size_each_group = 30\n"
            "method_a_scores = rng.normal(loc=71, scale=9, size=sample_size_each_group)\n"
            "method_b_scores = rng.normal(loc=77, scale=9, size=sample_size_each_group)\n\n"
            "students = pd.DataFrame({\n"
            "    'exam_score': np.concatenate([method_a_scores, method_b_scores]),\n"
            "    'method': ['Method A'] * sample_size_each_group + ['Method B'] * sample_size_each_group,\n"
            "})\n\n"
            "students.head()"
        ),
        md("### Step 4.2 — Short EDA"),
        code(
            "print('Sample size by method:')\n"
            "print(students['method'].value_counts())\n"
            "print('\\nDescriptive statistics:')\n"
            "students.groupby('method')['exam_score'].describe().round(2)"
        ),
        code(
            "plt.figure(figsize=(6, 4))\n"
            "students.boxplot(column='exam_score', by='method')\n"
            "plt.title('Exam scores by method')\n"
            "plt.suptitle('')\n"
            "plt.xlabel('Method')\n"
            "plt.ylabel('Exam score')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md("**Interpretation:** Method B appears higher on average, but sample variability remains visible in the boxplot."),
        md(
            "### Step 4.3 — State hypotheses\n\n"
            "- **H0:** Mean exam score is the same for Method A and Method B.\n"
            "- **H1:** Mean exam scores are different between methods."
        ),
        md("### Step 4.4 — One statistical test only (t-test)"),
        code(
            "a_scores = students.loc[students['method'] == 'Method A', 'exam_score']\n"
            "b_scores = students.loc[students['method'] == 'Method B', 'exam_score']\n\n"
            "t_result = stats.ttest_ind(a_scores, b_scores, equal_var=False)\n"
            "print('t statistic:', round(t_result.statistic, 3))\n"
            "print('p-value:', round(t_result.pvalue, 4))"
        ),
        md("### Step 4.5 — Confidence interval for mean difference"),
        code(
            "mean_diff = b_scores.mean() - a_scores.mean()\n"
            "se_diff = np.sqrt(a_scores.var(ddof=1)/len(a_scores) + b_scores.var(ddof=1)/len(b_scores))\n"
            "df_num = (a_scores.var(ddof=1)/len(a_scores) + b_scores.var(ddof=1)/len(b_scores)) ** 2\n"
            "df_den = ((a_scores.var(ddof=1)/len(a_scores))**2 / (len(a_scores)-1)) + ((b_scores.var(ddof=1)/len(b_scores))**2 / (len(b_scores)-1))\n"
            "degrees_freedom = df_num / df_den\n"
            "t_critical = stats.t.ppf(0.975, degrees_freedom)\n"
            "margin = t_critical * se_diff\n"
            "ci_low = mean_diff - margin\n"
            "ci_high = mean_diff + margin\n\n"
            "print('Mean difference (B - A):', round(mean_diff, 2))\n"
            "print('95% confidence interval: [', round(ci_low, 2), ',', round(ci_high, 2), ']')"
        ),
        md(
            "**Interpretation:** The confidence interval gives a range of plausible values for the true mean difference. "
            "If 0 is not inside the interval, that supports a difference."
        ),
        md("### Step 4.6 — Plain-language conclusion"),
        code(
            "alpha = 0.05\n"
            "if t_result.pvalue < alpha:\n"
            "    conclusion = (\n"
            "        'We reject H0 at alpha = 0.05. The data suggest Method A and Method B '\n"
            "        'have different average exam scores in this sample.'\n"
            "    )\n"
            "else:\n"
            "    conclusion = (\n"
            "        'We do not reject H0 at alpha = 0.05. The observed difference could be due to random sampling variation.'\n"
            "    )\n"
            "print(conclusion)"
        ),
        md("### Step 4.7 — Change sample size or group difference"),
        code(
            "small_sample_a = rng.normal(71, 9, 10)\n"
            "small_sample_b = rng.normal(77, 9, 10)\n"
            "_, p_small = stats.ttest_ind(small_sample_a, small_sample_b, equal_var=False)\n\n"
            "large_sample_a = rng.normal(71, 9, 200)\n"
            "large_sample_b = rng.normal(77, 9, 200)\n"
            "_, p_large = stats.ttest_ind(large_sample_a, large_sample_b, equal_var=False)\n\n"
            "size_effect = pd.DataFrame({\n"
            "    'sample_size_per_group': [10, 200],\n"
            "    'p_value': [p_small, p_large],\n"
            "}).round(4)\n"
            "size_effect"
        ),
        md("**Interpretation:** With larger samples, we usually get stronger evidence (smaller p-values) for the same underlying difference."),
    ]
    cells += footer(
        "Why should we avoid running many tests without a clear question?",
        "Each test has a chance of a false positive. Many unplanned tests increase the risk of finding 'significant' results by luck.",
        "Change `sample_size_each_group` from 30 to 10 and rerun the t-test. Observe how the p-value and confidence interval change.",
        "- Start with one focused question.\n"
        "- Explore data briefly before testing.\n"
        "- Report estimate + uncertainty, not only p-values.\n"
        "- Write conclusions in plain language.",
        "Rewrite the final conclusion in non-technical language for a classmate who has not taken statistics.",
        summary="We followed one clear question from EDA to a single t-test, confidence interval, and plain-language conclusion.",
        refinement1=(
            "Change `sample_size_each_group` from `30` to `10`, rebuild the dataset, and rerun the t-test. "
            "How do the p-value and confidence interval change?"
        ),
        refinement2=(
            "Keep sample size at 30, but make the methods more similar "
            "(for example, `method_a_scores = rng.normal(loc=75, ...)` and `method_b_scores = rng.normal(loc=76, ...)`). "
            "Rerun the test and explain whether you still reject H0."
        ),
    )
    cells.append(code("print('Notebook 05 completed successfully.')"))
    return cells
