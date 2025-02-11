import yfinance as yf
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random

# Function to fetch stock summary from Yahoo Finance
def fetch_yahoo_finance_summary(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    summary = info.get('longBusinessSummary', '')
    return summary

# Function to generate random financial data (tickers, prices, trade details)
def generate_random_financial_data():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
    prices = [150.25, 302.15, 2800.60, 3345.00, 895.50, 365.20]
    sedol = [f"{random.randint(1000000, 9999999)}" for _ in range(6)]
    cusip = [f"{random.randint(10000000, 99999999)}" for _ in range(6)]
    isin = [f"US{random.randint(1000000000, 9999999999)}" for _ in range(6)]

    ticker = random.choice(tickers)
    price = random.choice(prices)
    trade_sedol = random.choice(sedol)
    trade_cusip = random.choice(cusip)
    trade_isin = random.choice(isin)

    return ticker, price, trade_sedol, trade_cusip, trade_isin

# Load FinanceGPT model and tokenizer
model_name = "AI4Finance/FinanceGPT"  # Replace with actual model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a pipeline for text generation (question generation and answering)
fin_gpt_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to generate a dynamic question based on the summary
def generate_dynamic_question(summary):
    # Prepare the prompt for the model to generate a relevant question.
    context_for_question = f"Based on the following financial summary, generate a relevant finance-related question: {summary}"

    # Generate the question using FinanceGPT model
    generated_question = fin_gpt_pipeline(context_for_question, max_length=50, do_sample=True, temperature=0.7)
    
    return generated_question[0]['generated_text']

# Function to generate an answer to the question
def generate_answer(question):
    # Generate the answer using FinanceGPT model
    generated_answer = fin_gpt_pipeline(question, max_length=150, do_sample=True, temperature=0.7)
    
    return generated_answer[0]['generated_text']

# Generate random financial data
ticker, price, trade_sedol, trade_cusip, trade_isin = generate_random_financial_data()

# Fetch the Yahoo Finance summary for the selected ticker
summary = fetch_yahoo_finance_summary(ticker)

# Generate a dynamic question based on the summary
question = generate_dynamic_question(summary)

# Generate the answer to the question
answer = generate_answer(question)

# Append the random trade details to the generated answer
generated_answer = answer + f"\n\nTrade Details:\nTicker: {ticker}\nPrice: ${price}\n" \
                            f"SEDOL: {trade_sedol}\nCUSIP: {trade_cusip}\nISIN: {trade_isin}"

# Output the final response
print(f"Q: {question}")
print(f"A: {generated_answer}")
