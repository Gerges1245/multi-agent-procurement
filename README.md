# Multi-Agent Procurement Assistant

A CrewAI project that searches Egyptian retailers, extracts product information, evaluates offers against a company purchase request, and produces an English HTML report.

The five stages run sequentially: search (Tavily), collection, scraping (ScrapeGraph v2 REST API), comparison, and reporting (Gemini). The comparison checks price, specifications, condition, and quantity. Rankable offers receive a value score weighted by price (60%), RAM (20%), and SSD (20%).

## Setup (Windows PowerShell)

Open PowerShell in this folder. Python 3.11 was used for the live run.

    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    Copy-Item .env.example .env
    notepad .env

Fill in your Gemini, Tavily, and ScrapeGraph API keys in the local .env file. Edit company_context.json to change the purchase request. Keep .env private.

## Run

    .\.venv\Scripts\python.exe .\run_pipeline.py
    Invoke-Item .\reports\procurement_report.html

Live runs use API quota and overwrite the generated report. The included report is a snapshot of a completed run. Its two reviewed offers exceeded the unit budget, so neither qualified for ranking.

To rebuild the report from the bundled comparison without requesting new API data:

    py -3.11 .\report_agent.py --render-only --output-dir .\preview

test_scraping.py also provides the extraction schema imported by the scraping and comparison stages.
