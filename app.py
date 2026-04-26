import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from openai import OpenAI
import os

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="AI Market Advisor", layout="wide", page_icon="🏢")

# Securely load the OpenAI API Key from Streamlit Secrets or Environment Variables
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    
    /* KPI Metric Cards */
    [data-testid="stMetricValue"] {
        color: #58A6FF !important;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        color: #C9D1D9 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1C2128;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363D;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* AI Advisor Cards */
    .advisor-card { 
        background-color: #1C2128; 
        padding: 22px; 
        border-radius: 12px; 
        border: 1px solid #30363D;
        border-left: 6px solid #238636;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .advisor-card h3 {
        margin-top: 0;
        color: #F0F6FC;
        font-size: 1.3rem;
    }
    .advisor-card p {
        color: #8B949E;
        margin: 4px 0;
    }
    .advisor-card b {
        color: #C9D1D9;
    }
    
    /* Strategy Memo Section */
    .memo-container {
        background-color: #0D1117;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #388BFD;
        color: #C9D1D9;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA PROCESSING ENGINE
# ==========================================

@st.cache_data
def load_and_clean_data(file_name):
    # Use relative path so it works on GitHub and Streamlit Cloud
    try:
        # Check if the file exists in the current directory
        if not os.path.exists(file_name):
            st.error(f"Dataset file '{file_name}' not found in the project directory.")
            return pd.DataFrame()
            
        df_raw = pd.read_excel(file_name)
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return pd.DataFrame()
        
    df_raw.columns = [str(c).replace('\n', '') for c in df_raw.columns]
    
    # Map required HUD columns
    mapped_cols = {
        'HUD Fair Market Rent Area Name': 'Market',
        'SAFMR0BR': 'Studio_Rent',
        'SAFMR1BR': '1BR_Rent',
        'SAFMR2BR': '2BR_Rent',
        'SAFMR3BR': '3BR_Rent',
        'SAFMR4BR': '4BR_Rent'
    }
    
    df_subset = df_raw[[c for c in mapped_cols.keys() if c in df_raw.columns]].rename(columns=mapped_cols)
    
    # Aggregate to Market level
    df_agg = df_subset.groupby('Market').mean().round().reset_index()
    
    # Extract State
    df_agg['State'] = df_agg['Market'].str.extract(r',\s*([A-Z]{2})').fillna('Unknown')
    
    return df_agg

def calculate_shpi(df, weights={'premium': 0.4, 'affordability': 0.2, 'uni': 0.2, 'pop': 0.2}):
    """Calculates the Student Housing Profitability Index based on weighted metrics."""
    if df.empty: return df
    
    # Financial spreads
    df['Price_Gap'] = df['3BR_Rent'] - df['1BR_Rent']
    df['Roommate_Premium_Ratio'] = df['3BR_Rent'] / df['1BR_Rent']
    # Formula: abs((4BR Rent / 4) - Avg(0BR & 1BR Rent))
    df['Roommate_Premium'] = ((df['1BR_Rent'] * 4) - df['4BR_Rent']).clip(lower=0)
    
    # Mock Demographic Data
    np.random.seed(42)
    df['University_Count'] = np.random.randint(0, 12, size=len(df))
    df['Pop_18_29'] = np.random.randint(5000, 180000, size=len(df))
    
    # Normalization
    def norm(series):
        if series.max() == series.min(): return series * 0
        return (series - series.min()) / (series.max() - series.min())

    n_premium = norm(df['Roommate_Premium'])
    n_afford = 1 - norm(df['1BR_Rent']) 
    n_uni = norm(df['University_Count'])
    n_pop = norm(df['Pop_18_29'])
    
    # Weighted Score (0-100)
    df['SHPI_Score'] = (
        (n_premium * weights['premium']) + 
        (n_afford * weights['affordability']) + 
        (n_uni * weights['uni']) + 
        (n_pop * weights['pop'])
    ) * 100
    
    return df.round(2)

# ==========================================
# 3. AI ADVISOR ENGINE (LLM + LOGIC)
# ==========================================

def local_rule_expert(user_prompt, df):
    """Fallback logic that provides real data-driven insights when the LLM is unavailable."""
    if df.empty: return "Dataset is empty, cannot provide analysis."
    
    # Pre-calculate key metrics for the response
    top_market = df.nlargest(1, 'SHPI_Score').iloc[0]
    
    # Calculate top state by average SHPI
    state_scores = df.groupby('State')['SHPI_Score'].mean().sort_values(ascending=False)
    top_state = state_scores.index[0]
    top_state_score = state_scores.iloc[0]
    
    prompt_lower = user_prompt.lower()
    
    # 1. Handle State Queries
    if "state" in prompt_lower:
        return (f"**Local Data Advisor (Offline Mode):** \n\n"
                f"1. The state with the highest average SHPI score is **{top_state}** with an average score of {top_state_score:.2f}.\n"
                f"2. Within {top_state}, the top-performing market is {df[df['State']==top_state].nlargest(1, 'SHPI_Score').iloc[0]['Market']}.\n"
                f"3. Overall, the market with the single highest SHPI in the dataset is **{top_market['Market']}**.")

    # 2. Handle Risk/Danger Queries
    if "risk" in prompt_lower or "danger" in prompt_lower:
        risk_avg = df[df['SHPI_Score'] < 40]
        return (f"**Local Strategic Analysis (Offline Mode):** \n\n"
                f"1. Risk is highest in markets where 1BR rent is above ${df['1BR_Rent'].mean():.0f} but premiums are low.\n"
                f"2. There are {len(risk_avg)} markets identified as 'Low Potential' in your current view.\n"
                f"3. Avoid markets where the Roommate Premium is below the dataset median of ${df['Roommate_Premium'].median():.0f}.")
    
    # 3. Handle Best/Ranking Queries
    if "best" in prompt_lower or "rank" in prompt_lower or "highest" in prompt_lower:
        return (f"**Local Ranking Summary (Offline Mode):** \n\n"
                f"1. The #1 ranked market is **{top_market['Market']}** with a score of {top_market['SHPI_Score']}.\n"
                f"2. This market offers a monthly roommate premium of ${top_market['Roommate_Premium']:,.0f}.\n"
                f"3. Targeting **{top_state}** provides the best state-wide baseline for student housing development.")

    return (f"**Local Data Advisor (Offline Mode):** I am analyzing {len(df)} markets. "
            f"The current dataset shows an average Roommate Premium of ${df['Roommate_Premium'].mean():,.0f}. "
            f"Focus on **{top_state}** and **{top_market['Market']}** for immediate high-yield opportunities.")

def query_ai_market_expert(user_prompt, df):
    """Sends market data context to OpenAI and returns a strategic response with a local fallback."""
    if df.empty: return "Not enough data for analysis."
    
    # 1. ENHANCED CONTEXT: Provide statistical grounding and top outliers
    stats_summary = {
        "total_markets": len(df),
        "avg_premium": df['Roommate_Premium'].mean(),
        "avg_rent_1br": df['1BR_Rent'].mean(),
        "top_shpi_market": df.nlargest(1, 'SHPI_Score').iloc[0]['Market'],
        "highest_premium": df.nlargest(1, 'Roommate_Premium').iloc[0]['Market']
    }
    
    top_context = df.nlargest(12, 'SHPI_Score')[['Market', 'State', 'Roommate_Premium', 'SHPI_Score', '1BR_Rent']].to_string()
    
    # 2. REFINED SYSTEM PROMPT: Stricter rules for structure and accuracy
    system_msg = f"""You are an elite Real Estate Investment Consultant specializing in HUD Rent Data and Student Housing.
    DATASET SUMMARY: {stats_summary}
    TOP MARKET DATA: {top_context}
    
    STRICT INSTRUCTIONS:
    1. Every response MUST contain between 5 and 7 detailed bullet points.
    2. You MUST use specific numbers (Rent, SHPI, or Premium) from the data provided. Never guess.
    3. Analyze the specific 'Roommate Premium' spread (1BR x 4 vs. 4BR) as the primary ROI driver.
    4. Mention market risks (e.g. high entry cost) if applicable.
    5. Be critical, professional, and act as a senior decision-maker.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"User Strategic Question: {user_prompt}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return local_rule_expert(user_prompt, df)

def get_market_label(row):
    if row['SHPI_Score'] > 75 and row['Roommate_Premium'] > 1200:
        return "High Potential", "Green", "Yield Powerhouse: Strong spread and demographics."
    elif row['SHPI_Score'] > 50:
        return "Moderate Potential", "Orange", "Stable Growth: Consistent demand."
    else:
        return "Low Potential", "Red", "High Risk: Insufficient spreads."

def generate_ai_insights(df):
    if df.empty: return [], pd.DataFrame()
    top_5 = df.nlargest(5, 'SHPI_Score')
    bottom_5 = df.nsmallest(5, 'SHPI_Score')
    insights = []
    for _, row in top_5.iterrows():
        label, color, reason = get_market_label(row)
        insights.append({'market': row['Market'], 'score': row['SHPI_Score'], 'label': label, 'reason': reason, 'premium': row['Roommate_Premium']})
    return insights, bottom_5

# ==========================================
# 4. UI COMPONENTS (TABS)
# ==========================================

def render_overview(df):
    st.header("📍 Market Performance Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Roommate Premium", f"${df['Roommate_Premium'].mean():,.0f}")
    c2.metric("Top SHPI Score", f"{df['SHPI_Score'].max():.1f}")
    c3.metric("Markets Analyzed", len(df))
    c4.metric("Avg 1BR Rent", f"${df['1BR_Rent'].mean():,.0f}")
    st.divider()
    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(px.bar(df.nlargest(15, 'Roommate_Premium'), x='Roommate_Premium', y='Market', orientation='h', title="Top Markets by Premium ($)", color='Roommate_Premium'), use_container_width=True)
    with cb:
        st.plotly_chart(px.scatter(df, x='1BR_Rent', y='4BR_Rent', size='Roommate_Premium', color='SHPI_Score', title="Rent Correlation (Size=Premium)", hover_name='Market'), use_container_width=True)

def render_ai_advisor(df):
    st.header("🤖 AI Market Strategist")
    st.subheader("Interactive Strategy Consultation")
    user_query = st.text_input("Ask the expert (e.g., 'Summarize risks for high-rent markets'):")
    
    st.markdown("---")
    st.caption("💡 **Example Prompts:**")
    st.info("- *'Which top-ranked markets offer the best balance of low 1BR rent and high premium?'*\n- *'Identify the best entries for a developer with a $2000 4BR budget.'*\n- *'Why is the #1 ranked market considered a safer bet than the #5 ranked market?'*")
    st.markdown("---")
    
    if user_query:
        with st.spinner("AI Strategist is analyzing data..."):
            res = query_ai_market_expert(user_query, df)
            st.markdown(f"<div class='memo-container'><b>Strategy Memo:</b><br><br>{res}</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("🚀 Algorithmic Top Picks")
    insights, weak_markets = generate_ai_insights(df)
    for item in insights:
        st.markdown(f"<div class='advisor-card'><h3>{item['market']} — <span style='color:#238636'>{item['label']}</span></h3><p>Score: {item['score']} | Premium: ${item['premium']:,.0f}</p><p><i>Verdict: {item['reason']}</i></p></div>", unsafe_allow_html=True)

def render_scenario_simulator(df):
    st.header("⚙️ Scenario & Assumption Simulator")
    with st.expander("Adjust Global Weights", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        w_prem = c1.slider("Premium (%)", 0, 100, 40)
        w_aff = c2.slider("Affordability (%)", 0, 100, 20)
        w_uni = c3.slider("Universities (%)", 0, 100, 20)
        w_pop = c4.slider("Youth Pop (%)", 0, 100, 20)
        
    total = w_prem + w_aff + w_uni + w_pop
    weights = {'premium': w_prem/total, 'affordability': w_aff/total, 'uni': w_uni/total, 'pop': w_pop/total}
    
    sim_df = calculate_shpi(df.copy(), weights).sort_values('SHPI_Score', ascending=False)
    st.plotly_chart(px.line(sim_df.head(20), x='Market', y='SHPI_Score', title="Sensitivity Analysis: Top 20 Ranking Shifts", markers=True), use_container_width=True)
    st.success(f"Simulation Complete. Under current weights, **{sim_df.iloc[0]['Market']}** is the #1 priority.")

def main():
    st.sidebar.title("🎛️ Analysis Controls")
    data = load_and_clean_data("fy2026_safmrs.xlsx")
    if data.empty: return
    
    df = calculate_shpi(data)
    
    states = sorted(df['State'].unique())
    selected_states = st.sidebar.multiselect("Filter States", states, default=["TX", "CA", "FL", "NY"] if "TX" in states else states[:3])
    filtered_df = df[df['State'].isin(selected_states)] if selected_states else df
    
    rent_limit = st.sidebar.slider("Max 1BR Rent", int(df['1BR_Rent'].min()), int(df['1BR_Rent'].max()), int(df['1BR_Rent'].max()))
    filtered_df = filtered_df[filtered_df['1BR_Rent'] <= rent_limit]

    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🔍 Market Explorer", "📈 SHPI Analysis", "🤖 AI Advisor", "⚙️ Simulator"])
    
    with t1: render_overview(filtered_df)
    with t2: st.dataframe(filtered_df[['Market', 'State', '1BR_Rent', '4BR_Rent', 'Roommate_Premium', 'SHPI_Score']], use_container_width=True)
    with t3: st.plotly_chart(px.scatter(filtered_df, x="Pop_18_29", y="SHPI_Score", size="University_Count", color="Roommate_Premium", hover_name="Market", color_continuous_scale="Viridis", height=600), use_container_width=True)
    with t4: render_ai_advisor(filtered_df)
    with t5: render_scenario_simulator(filtered_df)

if __name__ == "__main__":
    main()
