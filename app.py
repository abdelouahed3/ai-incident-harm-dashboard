import streamlit as st
import pandas as pd
import plotly.express as px

st.title("AI Incident Harm Dashboard")
st.markdown("Exploring predictors of tangible harm in AI incidents from the AI Incident Database.")

@st.cache_data
def load_data():
    df = pd.read_csv("model_df.csv")
    df["minor_flag"] = (df["involving_minor"] == "yes").astype(int)
    df["rights_flag"] = (df["rights_violation"] == "yes").astype(int)
    df["physical_flag"] = (df["physical_objects"] == "yes").astype(int)
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df

df = load_data()

# Sidebar
st.sidebar.header("Filters")

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

domain_options = sorted(df["mit_risk_domain"].dropna().unique())
domains = st.sidebar.multiselect(
    "MIT Risk Domain",
    options=domain_options,
    default=domain_options
)

# Apply filters
filtered = df[
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1]) &
    (df["mit_risk_domain"].isin(domains))
]

# KPI row
col1, col2, col3 = st.columns(3)
col1.metric("Incidents shown", len(filtered))
col2.metric("Tangible harm incidents", int((filtered["harm_outcome"] == "TangibleHarm").sum()))
col3.metric("Physical objects involved", int(filtered["physical_flag"].sum()))

# Chart 1: Outcome distribution
st.subheader("Harm outcome distribution")

order = ["NoHarm", "Issue", "NearMiss", "TangibleHarm"]
outcome_counts = filtered["harm_outcome"].value_counts().reindex(order).reset_index()
outcome_counts.columns = ["Harm Outcome", "Count"]

fig = px.bar(
    outcome_counts,
    x="Harm Outcome",
    y="Count",
    color="Harm Outcome",
    color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    title=f"Harm outcomes for {len(filtered)} incidents"
)
fig.update_layout(showlegend=False, height=450)
st.plotly_chart(fig, use_container_width=True)

# Chart 2: Odds ratio forest plot (static, from the ordinal regression)
# Chart 2: Odds ratio forest plot (static, from the ordinal regression)
st.subheader("What predicts harm escalation?")
st.markdown(
    "Odds ratios from an ordinal logistic regression on 197 incidents. "
    "Values above 1 push incidents toward more severe harm categories. "
    "Bars crossing the dashed line are not statistically significant."
)

# Results from the fitted model (n=197) — see notebook for full output
forest_data = pd.DataFrame({
    "predictor": ["Involves physical objects", "Rights violation", "Involves minor"],
    "odds_ratio": [9.633, 7.699, 0.482],
    "ci_low": [5.033, 3.092, 0.169],
    "ci_high": [18.436, 19.170, 1.376],
    "p_value": [0.000, 0.000, 0.173],
})

fig2 = px.scatter(
    forest_data,
    x="odds_ratio",
    y="predictor",
    log_x=True,
    color="p_value",
    color_continuous_scale=["#C44E52", "#55A868"],
    size=[15, 15, 15],
    hover_data={"ci_low": ":.2f", "ci_high": ":.2f", "p_value": ":.3f"},
    title="Odds ratios with 95% confidence intervals"
)

# Draw confidence interval bars as horizontal lines
for i, row in forest_data.iterrows():
    fig2.add_shape(
        type="line",
        x0=row["ci_low"], x1=row["ci_high"],
        y0=i, y1=i,
        line=dict(color="#C44E52", width=3)
    )

# Reference line at OR = 1
fig2.add_vline(x=1, line_dash="dash", line_color="gray",
               annotation_text="No effect (OR = 1)",
               annotation_position="top")

fig2.update_layout(
    height=400,
    xaxis_title="Odds ratio (log scale)",
    yaxis_title="",
    xaxis=dict(range=[0.1, 30]),
    coloraxis_showscale=False,
)
st.plotly_chart(fig2, use_container_width=True)

# Interpretation box
st.info(
    "**Key finding:** Incidents involving physical objects had **9.6× higher odds** "
    "(95% CI: 5.0–18.4, p < 0.001) of being in a more severe harm category. "
    "Rights violations had **7.7× higher odds** (95% CI: 3.1–19.2, p < 0.001). "
    "Involvement of minors was **not significant** (OR = 0.48, p = 0.17)."
)

# Chart 3: Harm outcome by physical-object involvement (stacked bar)
st.subheader("Does physical involvement escalate harm?")
st.markdown(
    "Percentage of incidents in each harm category, split by whether physical objects were involved. "
    "If physical involvement drives harm, the red 'TangibleHarm' segment should be much larger on the right."
)

order = ["NoHarm", "Issue", "NearMiss", "TangibleHarm"]

# Compute within-group percentages
cross = pd.crosstab(
    filtered["physical_flag"],
    filtered["harm_outcome"],
    normalize="index"
) * 100

# Reindex to guarantee all four outcome columns exist (even if count = 0)
cross = cross.reindex(columns=order, fill_value=0)
cross = cross.reset_index().melt(
    id_vars="physical_flag",
    var_name="Outcome",
    value_name="Percent"
)

# Make physical_flag readable
cross["Physical Objects"] = cross["physical_flag"].map({0: "Not involved", 1: "Involved"})

fig3 = px.bar(
    cross,
    x="Physical Objects",
    y="Percent",
    color="Outcome",
    category_orders={"Outcome": order},
    color_discrete_sequence=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
    title="Harm outcome by physical-object involvement (% within group)",
    labels={"Percent": "% of incidents within group"},
    barmode="group" 
)
fig3.update_layout(
    height=450,
    yaxis=dict(range=[0, 75]),
    legend=dict(title="Harm outcome", orientation="h", y=-0.2)
)
st.plotly_chart(fig3, use_container_width=True)