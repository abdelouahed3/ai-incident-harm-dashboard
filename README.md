# AI Incident Harm Analysis

**What factors predict whether an AI incident escalates from a near-miss to tangible harm?**

End-to-end analysis of 990 real AI incidents from the [AI Incident Database](https://incidentdatabase.ai/), using Python, PostgreSQL, statsmodels, and Streamlit.

**Live dashboard:** *(added after deployment)*

---

## Key Finding

Ordinal logistic regression on 197 annotated incidents:

| Predictor | Odds Ratio | 95% CI | p |
|---|---|---|---|
| **Involves physical objects** | **9.63** | [5.03, 18.44] | < 0.001 |
| **Rights violation** | **7.70** | [3.09, 19.17] | < 0.001 |
| Involves minor | 0.48 | [0.17, 1.38] | 0.17 |

Physical-object incidents are ~9.6× more likely to escalate to tangible harm. The full model beats a null model (LR = 61.4, p < 0.001) and a multinomial logit on AIC (395.3 vs 499.5).

---

## Data & Method

- **Source:** 990 incidents scraped from the [AIID](https://incidentdatabase.ai/) via Apify
- **Storage:** PostgreSQL with 3 normalized tables (`incidents`, `taxonomy_cset`, `taxonomy_mit_gmf`)
- **Modeling subset:** 197 incidents with CSETv1 *Tangible Harm* labels (NoHarm → Issue → NearMiss → TangibleHarm)
- **Model:** Ordinal logistic regression with 3 binary predictors, plus LR test and AIC comparison vs multinomial logit
- **Dashboard:** Streamlit app with year and risk-domain filters, KPI metrics, and 3 charts

---

## Project Structure

```text
├── analysis.ipynb          # Full analysis: scraping → database → model
├── app.py                  # Streamlit dashboard
├── model_df.csv            # 197-row modeling dataset
├── raw_incidents_full.csv  # Raw scraped data (990 incidents)
├── requirements.txt
└── README.md
```

---

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
APIFY_TOKEN=your_token
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

Then run:

```bash
jupyter notebook analysis.ipynb    # analysis
streamlit run app.py               # dashboard at localhost:8501
```

---

## Limitations

- **20% label coverage:** Only 197/990 incidents have CSETv1 *Tangible Harm* labels (volunteer annotation process, not absence of harm)
- **Reporting bias:** AIID is built from media reports; high-profile incidents are over-represented
- **Descriptive, not causal:** Associations should not be read as causal mechanisms
- **Small sample:** n=197 limits power; CI on `physical_flag` is wide (5.0–18.4)

---

## Author

**Abdelouahed Gmiha** — MS in Data Science & Analytics  
[GitHub](https://github.com/abdelouahed3) · [LinkedIn](https://www.linkedin.com/in/abdelouahed-gmiha-450956274/)      