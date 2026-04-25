# 🏢 Roommate Premium & SHPI Analyzer
### Team 11 | Student Housing Strategy Tool

An AI-powered decision-support dashboard for real estate developers to identify the most profitable "dorm-style" apartment markets using HUD Fair Market Rent (FMR) data.

## 🚀 Features
- **Market Performance Overview:** Real-time visualization of rent spreads (1BR vs 4BR).
- **SHPI Scoring Engine:** A proprietary Student Housing Profitability Index (0-100) combining roommate premiums with demographic density.
- **🤖 AI Market Strategist:** An LLM-powered consultant that provides strategic investment memos based on current market data.
- **⚙️ Scenario Simulator:** Adjust development assumptions (weights for affordability, youth population, etc.) to see how market rankings shift.

## 🛠️ Deployment Instructions

### Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API Key:
   - Create a file at `.streamlit/secrets.toml`
   - Add your key: `OPENAI_API_KEY = "your-key-here"`
4. Run the app:
   ```bash
   streamlit run app.py
   ```

### Deploying to Streamlit Cloud
1. Push this folder to GitHub.
2. Connect your repo to [Streamlit Cloud](https://share.streamlit.io/).
3. In the Streamlit Cloud Dashboard, go to **Settings > Secrets** and paste your key:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

## 📊 Data Source
- **HUD SAFMR (2026):** Small Area Fair Market Rent dataset.
- **Internal Demographics:** Aggregated student population and university density data.
