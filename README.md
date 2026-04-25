# Roommate Premium & SHPI Analyzer

## Project Description
An AI-powered decision-support dashboard for real estate developers to identify the most profitable "dorm-style" apartment markets using HUD Fair Market Rent (FMR) data. The tool features a proprietary Student Housing Profitability Index (SHPI) and a Scenario Simulator to analyze market viability under different development assumptions.

## App Deployment URL

[YOUR_DEPLOYED_URL_HERE] *(e.g., https://roommate-analyzer.streamlit.app)*

## Local Setup Instructions

Ensure you have [uv](https://github.com/astral-sh/uv) installed on your machine.

```bash
# 1. Clone the repository
git clone [your-repository-url]

# 2. Navigate to the project folder
cd code

# 3. Synchronize dependencies and set up virtual environment
uv sync

# 4. Set up your API Key
# Create .streamlit/secrets.toml and add:
# OPENAI_API_KEY = "your-api-key-here"

# 5. Run the Streamlit app
uv run streamlit run app.py
```

## Project Structure
- `app.py`: Main Streamlit application.
- `fy2026_safmrs.xlsx`: HUD Fair Market Rent dataset.
- `pyproject.toml` & `uv.lock`: Dependency management via `uv`.
- `.streamlit/`: Local secrets configuration.
