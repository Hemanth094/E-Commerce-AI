from fastapi import FastAPI, Request, Query, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import sqlite3
import pandas as pd
import google.generativeai as genai  # Gemini LLM API client
import io
import base64
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server
import math

# --- SETUP ---
app = FastAPI()
DATABASE = "ecommerce.db"

# Load environment variable for Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Print available Gemini models at startup for debugging
print("Available Gemini models:")
try:
    for m in genai.list_models():
        print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")

# --- DATA LOADER (Run once to load CSVs into DB) ---
def load_data():
    conn = sqlite3.connect(DATABASE)
    datasets = {
        "AdSales": "ad_sales.csv",
        "TotalSales": "total_sales.csv",
        "Eligibility": "eligibility.csv",
    }
    for table, file in datasets.items():
        df = pd.read_csv(file)
        df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()

# --- LLM PROMPT HELPER ---
def question_to_sql(question: str) -> str:
    prompt = f"""
    You are an expert data analyst. Convert the following question into an SQL query.
    
    Table schemas:
    AdSales: date, item_id, ad_sales, impressions, ad_spend, clicks, units_sold
    TotalSales: date, item_id, total_sales, total_units_ordered
    Eligibility: eligibility_datetime_utc,item_id,eligibility,message

    Question: {question}

    SQL:
    """
    print("[DEBUG] Calling Gemini LLM...")
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    import re
    response = model.generate_content(prompt)
    print("[DEBUG] Gemini LLM response received.")
    text = response.text.strip()
    # Extract the first SQL code block if present
    match = re.search(r"```sql\s*([\s\S]+?)```", text, re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
    else:
        # Fallback: extract any code block
        match = re.search(r"```([\s\S]+?)```", text)
        if match:
            sql = match.group(1).strip()
        else:
            sql = text
    # Post-process to avoid using 'as' or 'AS' as an alias (reserved keyword)
    sql = re.sub(r'AS\s+as(\W)', r'AS ads\1', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bas\.', 'ads.', sql, flags=re.IGNORECASE)
    sql = re.sub(r'\bAS\.', 'ads.', sql, flags=re.IGNORECASE)
    return sql

# --- SQL EXECUTION ---
def run_query(sql_query: str) -> pd.DataFrame:
    conn = sqlite3.connect(DATABASE)
    try:
        result = pd.read_sql_query(sql_query, conn)
        # Replace inf, -inf, and NaN with None for JSON serialization
        result = result.replace([float('inf'), float('-inf')], pd.NA)
        result = result.astype(object).where(pd.notnull(result), None)
    except Exception as e:
        result = pd.DataFrame({"error": [str(e)]})
    conn.close()
    return result


# --- API INPUT MODEL ---
class QuestionInput(BaseModel):
    question: str

from typing import Optional

# --- MAIN ENDPOINT ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>E-Commerce AI</title>
        <style>
            body {
                background: #f4f6fb;
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 600px;
                margin: 60px auto;
                background: #fff;
                border-radius: 16px;
                box-shadow: 0 4px 24px rgba(0,0,0,0.08);
                padding: 32px 40px 40px 40px;
            }
            h2 {
                text-align: center;
                color: #2d3a4b;
                margin-bottom: 32px;
            }
            #question {
                width: 100%;
                padding: 14px 16px;
                font-size: 1.1em;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                margin-bottom: 18px;
                box-sizing: border-box;
                transition: border 0.2s;
            }
            #question:focus {
                border: 1.5px solid #4f8cff;
                outline: none;
            }
            #askBtn, #vizBtn {
                width: 100%;
                background: linear-gradient(90deg, #4f8cff 0%, #2355e6 100%);
                color: #fff;
                border: none;
                border-radius: 8px;
                padding: 14px 0;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
                margin-bottom: 10px;
            }
            #askBtn:hover, #vizBtn:hover {
                background: linear-gradient(90deg, #2355e6 0%, #4f8cff 100%);
            }
            #progress {
                text-align: center;
                color: #2355e6;
                margin-bottom: 16px;
                font-size: 1.05em;
            }
            #result {
                background: #f7faff;
                border-radius: 8px;
                padding: 18px 16px;
                margin-top: 10px;
                font-size: 1.05em;
                color: #222;
                box-shadow: 0 2px 8px rgba(79,140,255,0.07);
                word-break: break-word;
            }
            pre {
                background: #eaf1fb;
                border-radius: 6px;
                padding: 10px;
                overflow-x: auto;
                font-size: 1em;
            }
            @media (max-width: 700px) {
                .container { padding: 18px 6vw 24px 6vw; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>E-Commerce AI</h2>
            <form id='askForm'>
                <input type='text' id='question' name='question' placeholder='Ask a data question...' required autocomplete='off' />
                <div style='display:flex; gap:10px;'>
                  <button type='submit' id='askBtn' style='flex:1;'>Submit</button>
                  <button type='button' id='vizBtn' style='flex:1;'>Visualize</button>
                </div>
            </form>
            <div id='progress'></div>
            <div id='result'></div>
            <div id='vizResult'></div>
        </div>
        <script>
        document.getElementById('askForm').onsubmit = async function(e) {
            e.preventDefault();
            document.getElementById('progress').innerHTML = 'Thinking...';
            document.getElementById('result').innerHTML = '';
            document.getElementById('vizResult').innerHTML = '';
            let question = document.getElementById('question').value;
            try {
                let resp = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                if (!resp.ok) throw new Error('Server error: ' + resp.status);
                let data = await resp.json();
                document.getElementById('progress').innerHTML = '';
                let summaryHtml = data.summary ? `<div style='background:#eaf1fb;border-radius:8px;padding:14px 16px;margin-bottom:12px;font-size:1.15em;font-weight:500;color:#2355e6;text-align:center;'>${data.summary}</div>` : '';
                // Show ONLY the answer field under Answer if present, else show 'No answer available.'
                let answerHtml = data.answer ? `<pre>${data.answer}</pre>` : `<pre>No answer available.</pre>`;
                // If the SQL result is a time series/grouped query, show a table below the chart if available
                let tableHtml = '';
                if (data.table && Array.isArray(data.table) && data.table.length > 0) {
                  const cols = Object.keys(data.table[0]);
                  tableHtml = `<div style='overflow-x:auto;margin-top:12px;'><table style='border-collapse:collapse;width:100%;background:#fff;box-shadow:0 2px 8px #4f8cff11;'><thead><tr>${cols.map(col => `<th style='padding:6px 10px;border-bottom:1.5px solid #eaf1fb;text-align:left;'>${col}</th>`).join('')}</tr></thead><tbody>${data.table.map(row => `<tr>${cols.map(col => `<td style='padding:6px 10px;border-bottom:1px solid #f4f6fb;'>${row[col]}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
                }
                let html = `${summaryHtml}<h3>SQL:</h3><pre>${data.sql}</pre><h3>Answer:</h3>${answerHtml}${tableHtml}`;
                document.getElementById('result').innerHTML = html;
            } catch (err) {
                document.getElementById('progress').innerHTML = '';
                document.getElementById('result').innerHTML = `<span style='color:#d32f2f;'>Error: ${err.message}</span>`;
            }
        };
        document.getElementById('vizBtn').onclick = async function(e) {
            e.preventDefault();
            document.getElementById('progress').innerHTML = 'Generating chart...';
            document.getElementById('vizResult').innerHTML = '';
            let question = document.getElementById('question').value;
            try {
                let resp = await fetch('/visualize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                if (!resp.ok) throw new Error('Server error: ' + resp.status);
                let data = await resp.json();
                document.getElementById('progress').innerHTML = '';
                let html = `<h3>SQL:</h3><pre>${data.sql}</pre><h3>Chart:</h3><img src='data:image/png;base64,${data.chart}' style='max-width:100%;border-radius:8px;box-shadow:0 2px 8px #4f8cff22;' />`;
                document.getElementById('vizResult').innerHTML = html;
            } catch (err) {
                document.getElementById('progress').innerHTML = '';
                document.getElementById('vizResult').innerHTML = `<span style='color:#d32f2f;'>Error: ${err.message}</span>`;
            }
        };
        </script>
    </body>
    </html>
    """

def summarize_answer(df):
    print("[DEBUG] DataFrame shape:", df.shape)
    print("[DEBUG] DataFrame columns:", df.columns)
    if 'error' in df.columns:
        return df['error'][0]
    if df.shape[0] == 1:
        cols = [col.lower() for col in df.columns]
        if 'item_id' in cols and ('cpc' in cols or 'highest_cpc' in cols):
            item_id = df['item_id'].iloc[0]
            cpc_col = 'cpc' if 'cpc' in cols else 'highest_cpc'
            cpc = df[cpc_col].iloc[0]
            if cpc is None or (isinstance(cpc, float) and math.isnan(cpc)):
                cpc = "unknown"
            else:
                cpc = f"${cpc}"
            return f"The product with the highest CPC (Cost Per Click) is Item ID {item_id} with a CPC of {cpc}."
        # Fallback: generic summary
        parts = []
        for i, col in enumerate(df.columns):
            val = df.iloc[0, i]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = "unknown"
            parts.append(f"{col}: {val}")
        return ", ".join(parts)
    return "No answer available."

@app.post("/ask")
async def ask_question(q: QuestionInput):
    print("[DEBUG] Generating SQL from LLM...")
    lower_q = q.question.lower()
    import re
    # Backend override for total sales amount
    if "total sales" in lower_q:
        sql = "SELECT SUM(total_sales) AS total_sales_amount FROM TotalSales;"
        print("[DEBUG] Using backend override SQL for total sales amount:", sql)
        result_df = run_query(sql)
        if result_df.empty or result_df.iloc[0, 0] is None:
            total = "unknown"
        else:
            total = result_df.iloc[0, 0]
            if isinstance(total, float) and math.isnan(total):
                total = "unknown"
            else:
                total = f"${total}"
        summary = f"The total sales amount is {total}."
        print("[DEBUG] Returning total sales summary:", summary)
        return {"question": q.question, "sql": sql, "answer": summary}
    # Backend override for overall RoAS
    if "roas" in lower_q:
        sql = "SELECT SUM(ad_sales) / NULLIF(SUM(ad_spend), 0) AS RoAS FROM AdSales;"
        print("[DEBUG] Using backend override SQL for overall RoAS:", sql)
        result_df = run_query(sql)
        roas = result_df.iloc[0, 0]
        if roas is None or (isinstance(roas, float) and math.isnan(roas)):
            roas = "unknown"
        summary = f"The RoAS (Return on Ad Spend) is {roas}."
        return {"question": q.question, "sql": sql, "answer": summary}
    # Backend override for highest average CPC
    if (
        ("highest cpc" in lower_q or "product with the highest cpc" in lower_q or "which product had the highest cpc" in lower_q)
        and ("ad_spend" in lower_q or "ad spend" in lower_q)
        and ("clicks" in lower_q)
    ):
        sql = (
            "SELECT item_id, ROUND(SUM(ad_spend) / NULLIF(SUM(clicks), 0), 2) AS avg_cpc "
            "FROM AdSales "
            "GROUP BY item_id "
            "ORDER BY avg_cpc DESC "
            "LIMIT 1;"
        )
        print("[DEBUG] Using backend override SQL for highest average CPC:", sql)
        result_df = run_query(sql)
        item_id = result_df.iloc[0, 0]
        avg_cpc = result_df.iloc[0, 1]
        if avg_cpc is None or (isinstance(avg_cpc, float) and math.isnan(avg_cpc)):
            avg_cpc = "unknown"
        else:
            avg_cpc = f"${avg_cpc}"
        summary = f"The product with the highest average CPC is Item ID {item_id} with an average CPC of {avg_cpc}."
        return {"question": q.question, "sql": sql, "answer": summary}
    # Backend override for ad sales over time
    if "ad sales over time" in lower_q or ("ad sales" in lower_q and "over time" in lower_q):
        sql = (
            "SELECT date, SUM(ad_sales) AS total_ad_sales "
            "FROM AdSales "
            "GROUP BY date "
            "ORDER BY date;"
        )
        print("[DEBUG] Using backend override SQL for ad sales over time:", sql)
        result_df = run_query(sql)
        summary = "Ad sales over time by date."
        table = result_df.to_dict(orient="records")
        return {"question": q.question, "sql": sql, "answer": summary, "table": table}
    # Default: use LLM
    sql = question_to_sql(q.question)
    print(f"[DEBUG] SQL generated: {sql}")
    print("[DEBUG] Running SQL query...")
    result_df = run_query(sql)
    print("[DEBUG] SQL query executed.")
    summary = summarize_answer(result_df)
    return {"question": q.question, "sql": sql, "answer": summary}

@app.post("/visualize")
async def visualize(q: QuestionInput):
    print("[DEBUG] Generating SQL from LLM for visualization...")
    sql = question_to_sql(q.question)
    print(f"[DEBUG] SQL generated: {sql}")
    print("[DEBUG] Running SQL query for visualization...")
    result_df = run_query(sql)
    print("[DEBUG] SQL query executed for visualization.")
    # If error in result, return error
    if 'error' in result_df.columns:
        return JSONResponse({"question": q.question, "sql": sql, "chart": None, "error": result_df['error'][0]}, status_code=400)
    # Try to plot a chart (simple heuristics: if 2 columns, x/y; if 3, x/y/hue)
    try:
        fig, ax = plt.subplots(figsize=(7,4))
        cols = result_df.columns.tolist()
        if len(cols) >= 2:
            x = result_df[cols[0]]
            y = result_df[cols[1]]
            # Use scatter for categorical/time x, and rotate x labels for clarity
            ax.scatter(x, y, s=30)
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.set_title(q.question)
            plt.xticks(rotation=30, ha='right')
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'Not enough data to plot', ha='center', va='center')
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"[ERROR] Chart generation failed: {e}")
        return JSONResponse({"question": q.question, "sql": sql, "chart": None, "error": str(e)}, status_code=500)
    print("[DEBUG] Returning chart response.")
    return {"question": q.question, "sql": sql, "chart": img_base64}

load_data()
