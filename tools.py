# tools.py
import subprocess
import requests
import base64
from pathlib import Path
import sys
import json
import pandas as pd
from typing import List,Dict

import io
from sqlalchemy import create_engine, text
import openai
import wikipedia
import numpy as np
import tabula
from PIL import Image
import base64
from geopy.geocoders import Nominatim
from playwright.async_api import async_playwright
import asyncio
from bs4 import BeautifulSoup
import requests




TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetches the text content from a given URL. Use this for scraping websites or getting data from online sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The complete URL to fetch content from.",
                    },
                },
                "required": ["url"],
            },
        },                                                                      
    },
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": (
                "Executes Python code in an isolated environment for data analysis, manipulation, and visualization. "
                "The environment has pandas, matplotlib, numpy, and scikit-learn available. "
                "The code can access user-uploaded files directly by their filename (e.g., pd.read_csv('data.csv')). "
                "To return a plot, save it as 'output.png'. All print() output is captured as the result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute.",
                    },
                },
                "required": ["code"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "get_dataframe_info",
            "description": "Reads a data file (like a .csv or .parquet) and returns a JSON summary including column names, data types, non-null counts, and descriptive statistics (mean, std, min, max, etc.). This is the best first step for understanding any dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The filename of the data file to analyze (e.g., 'data.csv').",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "calculate_correlation",
            "description": "Computes the Pearson correlation coefficient between two specific numerical columns in a given data file. The name of this function is `calculate_correlation`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The filename of the data file (e.g., 'data.csv')."},
                    "column1": {"type": "string", "description": "The name of the first column."},
                    "column2": {"type": "string", "description": "The name of the second column."},
                },
                "required": ["file_path", "column1", "column2"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "create_pivot_table",
            "description": "Generates a pivot table to summarize data. This function takes a file and the names of the columns to use for the index, columns, and values of the pivot table. The name of this function is `create_pivot_table`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The filename of the data file (e.g., 'data.csv')."},
                    "index": {"type": "string", "description": "The name of the column to use as the pivot table's index (rows)."},
                    "columns": {"type": "string", "description": "The name of the column to use as the pivot table's columns."},
                    "values": {"type": "string", "description": "The name of the column to aggregate as the values in the pivot table."},
                },
                "required": ["file_path", "index", "columns", "values"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Executes a SQL query against a database (like SQLite or DuckDB) and returns the result as JSON. The name of this function is `run_sql_query`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute.",
                    },
                    "db_connection_string": {
                        "type": "string",
                        "description": "The SQLAlchemy connection string for the database. For an uploaded SQLite file named 'my_db.db', use 'sqlite:///my_db.db'. For a DuckDB file, use 'duckdb:///my_db.duckdb'.",
                    },
                },
                "required": ["query", "db_connection_string"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "get_sentiment",
            "description": "Analyzes a piece of text (like a movie review) to determine if its sentiment is positive, negative, or neutral. The name of this function is `get_sentiment`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_to_analyze": {
                        "type": "string",
                        "description": "The text content to be analyzed.",
                    },
                },
                "required": ["text_to_analyze"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "scrape_wikipedia_summary",
            "description": "Fetches the clean text summary from a Wikipedia page. Use this tool specifically for getting information from Wikipedia. The name of this function is `scrape_wikipedia_summary`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The title or search query for the Wikipedia page (e.g., 'Python (programming language)').",
                    },
                },
                "required": ["query"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "scrape_pdf_tables",
            "description": "Extracts all tabular data from a PDF document and returns it as a list of JSON objects. Use this for any PDF that contains tables. The name of this function is `scrape_pdf_tables`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The filename of the PDF file to process (e.g., 'report.pdf').",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "analyze_image_content",
            "description": "Analyzes an uploaded image file (e.g., a PNG or JPG) and answers a specific question about its contents. Use this to identify objects, read text, or describe scenes in an image. The name of this function is `analyze_image_content`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "The filename of the image to analyze (e.g., 'chart.png')."},
                    "prompt": {"type": "string", "description": "The specific question to ask about the image (e.g., 'What is the title of this chart?', 'Is there a cat in this picture?')."},
                },
                "required": ["image_path", "prompt"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "geocode_address",
            "description": "Finds the geographic coordinates (latitude and longitude) for a given street address, city, or landmark. Uses the Nominatim service. The name of this function is `geocode_address`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "The address or place name to geocode (e.g., '1600 Amphitheatre Parkway, Mountain View, CA' or 'Tokyo Tower').",
                    },
                },
                "required": ["address"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "scrape_dynamic_site",
            "description": 
"Renders a JavaScript-heavy website and saves the complete HTML to a file named 'scraped_page.html'. This is the first step in a two-step process. After calling this, use the 'parse_html' tool to extract specific data from the saved file. The name of this function is `scrape_dynamic_site`.",            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the dynamic website to scrape."},
                },
                "required": ["url"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "parse_html",
        "description": "Extracts specific data from an HTML file (like one saved by 'scrape_dynamic_site') using CSS selectors. Provide a dictionary where keys are desired data names and values are the CSS selectors to find that data. The name of this function is `parse_html`.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The local filename of the HTML file to parse (e.g., 'scraped_page.html')."},
                "selectors": {
                    "type": "object",
                    "description": "A JSON object of 'data_name': 'css_selector' pairs. For example: {\"titles\": \"h2.product-title\", \"prices\": \".price-tag\"}",
                },
            },
            "required": ["file_path", "selectors"],
        },
    },
},
{
    "type": "function",
    "function": {
        "name": "get_bbc_weather",
        "description": "Fetches the weather forecast for a location using its BBC Weather ID. Can provide a 3-day summary or a detailed hour-by-hour forecast. The name of this function is `get_bbc_weather`.",
        "parameters": {
            "type": "object",
            "properties": {
                "location_id": {
                    "type": "string",
                    "description": "The numerical ID for the location (e.g., '2643743' for London).",
                },
                "report_type": {
                    "type": "string",
                    "description": "The type of report to generate. Use 'summary' for a 3-day overview or 'detailed' for an hour-by-hour forecast.",
                    "enum": ["summary", "detailed"], # 'enum' helps the LLM choose a valid option
                },
            },
            "required": ["location_id"],
        },
    },
},








]


def get_bbc_weather(location_id: str, report_type: str = 'summary') -> str:
    """
    Fetches the weather forecast for a given BBC Weather location ID.
    Can return a 'summary' (default) or a 'detailed' hour-by-hour report.
    """
    print(f"Executing Tool 'get_bbc_weather' for ID: {location_id}, Type: {report_type}")
    
    url = f"https://weather-broker-cdn.api.bbci.co.uk/en/forecast/aggregated/{location_id}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        weather_data = response.json()
        
        forecasts_data = weather_data.get("forecasts", [])
        if not forecasts_data:
            return "Error: Forecast data not found in the API response."
        
        report = forecasts_data[0]
        location_name = report.get("location", {}).get("name")
        
        # --- NEW LOGIC ---
        if report_type == 'detailed':
            # Extract the detailed, timeseries forecast
            detailed_forecast = {
                "location_name": location_name,
                "issued_at": report.get("issuedAt"),
                "detailed_forecast": []
            }
            for slot in report.get("detailed", {}).get("reports", []):
                hour_summary = {
                    "timestamp": slot.get("localDate"),
                    "temperature_c": slot.get("temperatureC"),
                    "feels_like_temp_c": slot.get("feelsLikeTempC"),
                    "wind_speed_mph": slot.get("windSpeedMph"),
                    "wind_direction": slot.get("windDirectionAbbreviation"),
                    "precipitation_probability_percent": slot.get("precipitationProbabilityInPercent"),
                    "weather_type": slot.get("weatherType")
                }
                detailed_forecast["detailed_forecast"].append(hour_summary)
            return json.dumps(detailed_forecast, indent=2)

        else: # Default to 'summary'
            # The existing summary logic
            summary_report = {
                "location_name": location_name,
                "issued_at": report.get("issuedAt"),
                "daily_summary": []
            }
            for day in report.get("summary", {}).get("reports", []):
                day_summary = {
                    "date": day.get("localDate"),
                    "condition": day.get("weatherType"),
                    "max_temp_c": day.get("maxTempC"),
                    "min_temp_c": day.get("minTempC"),
                }
                summary_report["daily_summary"].append(day_summary)
            return json.dumps(summary_report, indent=2)

    except Exception as e:
        return f"An error occurred while processing weather data. Error: {e}"


def parse_html(file_path: str, selectors: Dict[str, str], work_dir: str) -> str:
    """
    Parses a local HTML file and extracts data using a dictionary of CSS selectors.
    For each key-value pair in the selectors dictionary, it finds elements matching
    the selector (value) and stores their text content under the given key.
    """
    print(f"Executing Tool 'parse_html' for file: {file_path}")
    full_path = Path(work_dir) / file_path
    if not full_path.exists():
        return f"Error: HTML file not found at {full_path}"

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "lxml")
        
        extracted_data = {}
        for data_key, selector in selectors.items():
            # Find all elements matching the selector
            elements = soup.select(selector)
            # Extract the text from each element, stripping whitespace
            extracted_data[data_key] = [el.get_text(strip=True) for el in elements]
            
        return json.dumps(extracted_data, indent=2)

    except Exception as e:
        return f"Failed to parse HTML file {file_path}. Error: {e}"


async def scrape_dynamic_site(url: str, work_dir: str) -> str:
    """
    Renders a JavaScript-heavy website using a headless browser and saves the
    complete, final HTML to a file named 'scraped_page.html'.
    """
    print(f"Executing Tool 'scrape_dynamic_site' for url: {url}")
    save_path = Path(work_dir) / "scraped_page.html"

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle', timeout=30000) # 30s timeout
            content = await page.content()
            await browser.close()
        
        # Save the full HTML content to the specified file
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Return a success message with the path to the saved file
        return json.dumps({
            "status": "success",
            "url": url,
            "saved_to": str(save_path.name) # Return just the filename
        })

    except Exception as e:
        return f"Failed to scrape dynamic site {url}. Error: {e}"


def geocode_address(address: str) -> str:
    """
    Converts a physical address or place name into geographic coordinates (latitude and longitude).
    """
    print(f"Executing Tool 'geocode_address' for address: {address}")
    try:
        # Create a geolocator instance. A unique user_agent is good practice.
        geolocator = Nominatim(user_agent="data_analyst_agent_v1")
        
        location = geolocator.geocode(address)
        
        if location is None:
            return f"Error: Could not find coordinates for the address '{address}'."
            
        result = {
            "address": address,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "full_address_found": location.address
        }
        
        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Failed to geocode address. Error: {e}"



def analyze_image_content(image_path: str, prompt: str, work_dir: str, client: openai.Client) -> str:
    """
    Analyzes the content of an image file using a multimodal LLM and answers a question about it.
    """
    print(f"Executing Tool 'analyze_image_content' for file: {image_path}")
    full_path = Path(work_dir) / image_path
    if not full_path.exists():
        return f"Error: Image file not found at {full_path}"

    try:
        # Open the image to verify it's a valid image file (optional but good practice)
        Image.open(full_path)

        # Encode the image to base64
        with open(full_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Call the multimodal model
        response = client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=500, # Allow for a reasonably detailed description
        )
        description = response.choices[0].message.content
        return json.dumps({"image": image_path, "analysis": description})

    except Exception as e:
        return f"Failed to analyze image. Error: {e}"


def scrape_wikipedia_summary(query: str) -> str:
    """
    Fetches the summary section of a Wikipedia page based on a search query.
    """
    print(f"Executing Tool 'scrape_wikipedia_summary' for query: {query}")
    try:
        # Fetch the summary of the page
        summary = wikipedia.summary(query, auto_suggest=True)
        
        result = {
            "query": query,
            "summary": summary
        }
        return json.dumps(result, indent=2)
        
    except wikipedia.exceptions.PageError:
        return f"Error: Could not find a Wikipedia page for the query '{query}'."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Error: The query '{query}' is ambiguous. It could refer to any of the following: {e.options}"
    except Exception as e:
        return f"Failed to scrape Wikipedia. Error: {e}"


def scrape_pdf_tables(file_path: str, work_dir: str) -> str:
    """
    Extracts all tables from a specified page in a PDF file.
    """
    print(f"Executing Tool 'scrape_pdf_tables' for file: {file_path}")
    full_path = Path(work_dir) / file_path
    if not full_path.exists():
        return f"Error: PDF file not found at {full_path}"

    try:
        # read_pdf returns a list of DataFrames, one for each table found
        tables_as_dfs = tabula.read_pdf(full_path, pages='all', multiple_tables=True)
        
        if not tables_as_dfs:
            return "No tables were found in the PDF file."
            
        # Convert each DataFrame in the list to a JSON string
        tables_as_json = [df.to_json(orient='split') for df in tables_as_dfs]
        
        # Return a JSON object containing the list of tables
        return json.dumps({"file_name": file_path, "extracted_tables": tables_as_json})

    except Exception as e:
        return f"Failed to scrape tables from PDF. Make sure Java is installed on the system. Error: {e}"


def get_sentiment(text_to_analyze: str, client: openai.Client) -> str:
    """
    Analyzes the sentiment of a given piece of text.
    """
    print(f"Executing Tool 'get_sentiment'")
    
    try:
        # We use a specific, constrained prompt to force the LLM to be a classifier
        response = client.chat.completions.create(
            model="openai/gpt-5-nano", # Use a fast and cheap model for this simple task
            messages=[
                {"role": "system", "content": "You are a sentiment analysis tool. Classify the user's text as 'positive', 'negative', or 'neutral'. Respond with only one of these three words and nothing else."},
                {"role": "user", "content": text_to_analyze}
            ],
            max_tokens=5, # Limit the output to a single word
            temperature=0.0 # Make the output deterministic
        )
        sentiment = response.choices[0].message.content.lower().strip()
        
        # Basic validation
        if sentiment not in ["positive", "negative", "neutral"]:
            return "Error: Could not determine a valid sentiment."
            
        return json.dumps({"text": text_to_analyze, "sentiment": sentiment})

    except Exception as e:
        return f"Failed to get sentiment. Error: {e}"


def run_sql_query(query: str, db_connection_string: str) -> str:
    """
    Executes a SQL query against a specified database and returns the result.
    Supports file-based databases like SQLite and DuckDB.
    For SQLite, the connection string should be 'sqlite:///path/to/database.db'.
    The path should be relative to the agent's working directory.
    """
    print(f"Executing Tool 'run_sql_query'")
    
    try:
        # Create a database engine from the connection string
        engine = create_engine(db_connection_string)
        
        # Execute the query and fetch results into a pandas DataFrame
        with engine.connect() as connection:
            result_df = pd.read_sql_query(sql=text(query), con=connection)
            
        # Return the result as a JSON string
        return result_df.to_json(orient="records")

    except Exception as e:
        return f"Failed to execute SQL query. Error: {e}"


def calculate_correlation(file_path: str, column1: str, column2: str, work_dir: str) -> str:
    """
    Calculates the Pearson correlation coefficient between two specified columns in a data file.
    """
    print(f"Executing Tool 'calculate_correlation' for file: {file_path}")
    full_path = Path(work_dir) / file_path
    if not full_path.exists():
        return f"Error: Data file not found at {full_path}"

    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(full_path)
        elif file_path.lower().endswith('.parquet'):
            df = pd.read_parquet(full_path)
        else:
            return f"Error: Unsupported file type."

        # Ensure columns exist
        if column1 not in df.columns or column2 not in df.columns:
            return f"Error: One or both columns ('{column1}', '{column2}') not found in the file."

        # Calculate correlation
        correlation = df[column1].corr(df[column2])
        
        result = {
            "file_name": file_path,
            "column_1": column1,
            "column_2": column2,
            "pearson_correlation": correlation
        }
        
        return json.dumps(result, indent=2)

    except Exception as e:
        return f"Failed to calculate correlation. Error: {e}"

def create_pivot_table(file_path: str, index: str, columns: str, values: str, work_dir: str) -> str:
    """
    Creates a pivot table from the data in the specified file.
    """
    print(f"Executing Tool 'create_pivot_table' for file: {file_path}")
    full_path = Path(work_dir) / file_path
    if not full_path.exists():
        return f"Error: Data file not found at {full_path}"

    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(full_path)
        elif file_path.lower().endswith('.parquet'):
            df = pd.read_parquet(full_path)
        else:
            return f"Error: Unsupported file type."

        # Create the pivot table
        pivot_table = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc=np.sum)
        
        # Return the pivot table as a JSON string
        return pivot_table.to_json(orient="split")

    except Exception as e:
        return f"Failed to create pivot table. Error: {e}"

def get_dataframe_info(file_path: str, work_dir: str) -> str:
    """
    Reads a data file (CSV, Parquet) and returns a summary of its contents.
    The summary includes column names, data types, and basic statistics.
    """
    print(f"Executing Tool 'get_dataframe_info' for file: {file_path}")
    full_path = Path(work_dir) / file_path
    if not full_path.exists():
        return f"Error: Data file not found at {full_path}"

    try:
        if file_path.lower().endswith('.csv'):
            df = pd.read_csv(full_path)
        elif file_path.lower().endswith('.parquet'):
            df = pd.read_parquet(full_path)
        else:
            return f"Error: Unsupported file type. Only .csv and .parquet are supported."

        # Use a string buffer to capture the output of df.info()
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)
        info_str = info_buffer.getvalue()

        # Get the statistical summary
        describe_df = df.describe(include='all')
        
        # Combine everything into a single, informative string
        summary = {
            "file_name": file_path,
            "info": info_str,
            "statistical_summary": describe_df.to_json(orient="split")
        }
        
        return json.dumps(summary, indent=2)

    except Exception as e:
        return f"Failed to get DataFrame info. Error: {e}"

def fetch_url(url: str) -> str:
    """Fetches text content from a specified URL using the AI Pipe proxy."""
    print(f"Executing Tool 'fetch_url' with URL: {url}")
    try:
        proxy_url = f"https://aipipe.org/proxy/{url}"
        response = requests.get(proxy_url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        return f"Error: Failed to fetch URL {url}. Reason: {e}"

def python_interpreter(code: str, work_dir: str) -> str:
    """
    Executes Python code in a sandboxed subprocess within a specific working directory.
    
    The code can access any files within its `work_dir`.
    If the code generates 'output.png', it will be base64 encoded and returned.
    """
    python_executable = sys.executable

    print(f"Executing Tool 'python_interpreter' in directory: {work_dir}")
    work_path = Path(work_dir)
    script_path = work_path / "agent_script.py"
    plot_path = work_path / "output.png"

    with open(script_path, "w") as f:
        f.write(code)
    print("\n\n--- 📜 DECODING: SCRIPT TO EXECUTE 📜 ---")
    print(code)
    print("------------------------------------------\n")

    try:
        python_executable = sys.executable
        
        # +++ ADD THIS DEBUG LINE +++
        print(f"--- [DEBUG] EXECUTING SUBPROCESS WITH PYTHON FROM: {python_executable} ---")
        # +++++++++++++++++++++++++++
        
        process = subprocess.run(
            [python_executable, str(script_path)],
            cwd=work_path, # Run the script from within the temp directory
            capture_output=True,
            text=True,
            timeout=1000,
            check=False
        )
        print("\n\n--- 📤 DECODING: SCRIPT RAW OUTPUT 📤 ---")
        print(f"Return Code: {process.returncode}")
        print("--- STDOUT ---")
        print(process.stdout)
        print("--- STDERR ---")
        print(process.stderr)
        print("------------------------------------------\n")
        if process.returncode != 0:
            return f"SCRIPT FAILED with return code {process.returncode}:\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
        stdout = process.stdout
        
        # Check if a plot was generated as output.png
        if plot_path.exists():
            with open(plot_path, "rb") as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            # Prepend the plot's data URI to the stdout
            plot_uri = f"data:image/png;base64,{img_base64}"
            return f"image_output:\n{plot_uri}\n\ntext_output:\n{stdout}"

        # If successful, just return the standard output
        return process.stdout


        

    except subprocess.CalledProcessError as e:
        return f"SCRIPT FAILED:\n--- STDOUT ---\n{e.stdout}\n--- STDERR ---\n{e.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: The Python script took too long to execute."
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    

    