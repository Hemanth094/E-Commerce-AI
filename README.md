# E-Commerce AI

A modern FastAPI-based web application for interactive data analysis and visualization of e-commerce metrics using natural language queries and Gemini LLM.

## Features
- **Natural Language Q&A:** Ask questions about your e-commerce data (sales, ad spend, RoAS, CPC, etc.) and get instant answers.
- **Automatic SQL Generation:** Converts your questions into SQL queries using Gemini LLM.
- **Data Visualization:** Visualize results as charts with a single click.
- **Smart Summaries:** Answers are returned in clear, human-readable language (not raw JSON).
- **Robust Error Handling:** Handles missing data, division by zero, and SQL errors gracefully.
- **Modern UI/UX:** Clean, responsive interface with prominent answers and easy-to-use controls.

## Technologies Used
- **FastAPI** (Python backend)
- **Pandas** (data manipulation)
- **Matplotlib** (charting)
- **SQLite** (local database)
- **Google Gemini LLM** (natural language to SQL)
- **HTML/CSS/JS** (frontend)

## Setup Instructions
1. **Clone the repository:**
   ```sh
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up your Gemini API key:**
   - Create a `.env` file with:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     ```
4. **Prepare your data:**
   - Place your CSV files (`ad_sales.csv`, `total_sales.csv`, `eligibility.csv`) in the project root.
   - The app will load them into `ecommerce.db` on startup.
5. **Run the app:**
   ```sh
   uvicorn main:app --reload
   ```
6. **Open in your browser:**
   - Go to [http://localhost:8000](http://localhost:8000)

## Usage
- Type a question (e.g., "What is the total sales amount?" or "Calculate the RoAS (Return on Ad Spend)") and click **Submit**.
- Click **Visualize** to see a chart for your query.
- Answers and charts are generated automatically from your data.

## Example Questions
- What is the total sales amount?
- Calculate the RoAS (Return on Ad Spend)
- Which product had the highest CPC?
- Show ad sales over time.

## License
MIT 
