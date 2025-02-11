import random
import yfinance as yf
from faker import Faker
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize Faker for random name and email generation
fake = Faker()

# Load model and tokenizer (for generating financial summaries, etc.)
model_name = "Ichabchiu/Llama-3.1-Omni-FinAI-8B"  # Example model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
fin_gpt_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to fetch stock summary from Yahoo Finance
def fetch_yahoo_finance_summary(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    summary = info.get('longBusinessSummary', '')
    return summary

# Function to generate random email sender and receiver
def generate_random_email():
    sender_name = fake.name()
    receiver_name = fake.name()
    sender_email = f"{sender_name.split()[0].lower()}.{sender_name.split()[1].lower()}@{random.choice(['gmail.com', 'yahoo.com', 'company.com'])}"
    receiver_email = f"{receiver_name.split()[0].lower()}.{receiver_name.split()[1].lower()}@{random.choice(['gmail.com', 'yahoo.com', 'company.com'])}"
    return sender_email, receiver_email

# Function to generate a random subject
def generate_random_subject():
    subjects = [
        "Financial Update Report",
        "Quarterly Earnings Call Summary",
        "Stock Performance Overview",
        "Market Update and Insights",
        "Important Investment Information"
    ]
    return random.choice(subjects)

# Function to generate random financial data
def generate_random_financial_data():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META']
    ticker = random.choice(tickers)
    return ticker

# Function to generate an email body using a financial summary
def generate_email_body(summary):
    context_for_email = f"Based on the following financial summary, please generate an email: {summary}"
    generated_email_body = fin_gpt_pipeline(context_for_email, max_length=150, do_sample=True, temperature=0.7)
    return generated_email_body[0]['generated_text']

# Example usage:
# Generate random financial data (stock ticker)
ticker = generate_random_financial_data()

# Fetch the Yahoo Finance summary for the selected ticker
summary = fetch_yahoo_finance_summary(ticker)

# Generate a random sender and receiver email addresses
sender_email, receiver_email = generate_random_email()

# Generate the email subject
subject = generate_random_subject()

# Generate the email body using the financial summary
email_body = generate_email_body(summary)

# Format the email
email = f"From: {sender_email}\nTo: {receiver_email}\nSubject: {subject}\n\n{email_body}\n\nBest regards,\n{sender_email}"

# Output the generated email
print(email)
