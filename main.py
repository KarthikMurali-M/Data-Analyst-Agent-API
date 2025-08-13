import re
from fastapi import FastAPI, UploadFile, File, HTTPException, status,Request
from typing import List,Dict
from pathlib import Path
import os
import openai
import tempfile  # <-- Import tempfile
import json
from tools import fetch_url, python_interpreter,get_dataframe_info ,calculate_correlation,create_pivot_table,run_sql_query,get_sentiment,scrape_wikipedia_summary,scrape_pdf_tables ,analyze_image_content ,geocode_address ,scrape_dynamic_site , parse_html ,get_bbc_weather ,TOOL_DEFINITIONS
import asyncio
import re
import time
# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()

if "AIPIPE_TOKEN" not in os.environ:
    raise RuntimeError("The AIPIPE_TOKEN environment variable is not set. Please set it to your token from aipipe.org.")

# Configure the OpenAI client to point to the AI Pipe proxy.
# The client will automatically use the OPENAI_API_KEY environment variable,
# so we set it to our AI Pipe token.
client = openai.OpenAI(
    base_url="https://aipipe.org/openrouter/v1",
    api_key=os.getenv("AIPIPE_TOKEN"),
)

AVAILABLE_TOOLS: Dict[str, callable] = {
    "fetch_url": fetch_url,
    "python_interpreter": python_interpreter,
    "get_dataframe_info": get_dataframe_info,
    "calculate_correlation": calculate_correlation,
    "create_pivot_table": create_pivot_table,
    "run_sql_query": run_sql_query,
    "get_sentiment": get_sentiment,
    "scrape_wikipedia_summary": scrape_wikipedia_summary,
    "scrape_pdf_tables": scrape_pdf_tables,
    "analyze_image_content": analyze_image_content,
    "geocode_address": geocode_address,
    "scrape_dynamic_site": scrape_dynamic_site,
    "parse_html": parse_html,
    "get_bbc_weather": get_bbc_weather
}


def is_output_valid(result: str | None) -> bool:
    """A simple validator to check if the agent's output is complete."""
    if result is None or result.strip() == "":
        return False
    try:
        data = json.loads(result)
        # Check for common failure indicators like null values or "N/A"
        if isinstance(data, list) and any(x is None or "N/A" in str(x) for x in data):
            return False
        if isinstance(data, dict) and any(v is None for v in data.values()):
            return False
    except (json.JSONDecodeError, TypeError):
        return False # Not valid JSON
    return True


app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses LLMs to source, prepare, analyze, and visualize data.  ;By 24f2001293@ds.study.iitm.ac.in",
)

@app.get("/")
def health_check():
    """A simple endpoint that the cloud platform can call to check if the service is alive."""
    return {"status": "ok", "message": "Data Analyst Agent is running."}

@app.post("/api/")
async def process_analysis_request(request: Request):
    # This is the main function that will be rewritten.

    max_retries = 3
    last_error = ""
    last_plan = {}
    
    with tempfile.TemporaryDirectory() as work_dir:
        work_path = Path(work_dir)
        form_data = await request.form()
        all_uploaded_files = [v for v in form_data.values() if hasattr(v, "filename")]
        
        if not all_uploaded_files:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files were uploaded.")

        # Read all discovered files into memory ONCE
        file_contents = {f.filename: await f.read() for f in all_uploaded_files}

        # --- Two-Pass Logic to Reliably Find the Question File ---
        all_filenames = list(file_contents.keys())
        questions_file_name = None
        first_txt_file_name = None

        # First pass: find the best candidate for the question file
        for filename in all_filenames:
            if filename.lower().endswith('.txt'):
                if first_txt_file_name is None:
                    first_txt_file_name = filename
                if 'question' in filename.lower():
                    questions_file_name = filename
                    break # Found the best possible match, stop looking

        # If no specific 'question.txt' was found, use the first .txt file as a fallback
        if questions_file_name is None:
            questions_file_name = first_txt_file_name
        
        if not questions_file_name: raise HTTPException(status.HTTP_400_BAD_REQUEST, "No .txt question file found.")
        
        # The task content is from the identified question file
        task_content = file_contents[questions_file_name].decode("utf-8")
        
        # The attachments are ALL files EXCEPT the question file. This is the key.
        attached_file_names = [f for f in all_filenames if f != questions_file_name]
        
        # Save all files to the temporary directory from memory
        for filename, content in file_contents.items():
            file_path = work_path / filename
            with open(file_path, "wb") as f: f.write(content)
        # ----------------------------------------------------
        time.sleep(0.1)
        # --- SELF-CORRECTION LOOP ---
        for i in range(max_retries):
            print(f"\n--- AGENT ATTEMPT #{i + 1} ---")

            # --- 2. PLANNING ---
            AVAILABLE_TOOLS: Dict[str, callable] = { "fetch_url": fetch_url, "python_interpreter": python_interpreter, "get_dataframe_info": get_dataframe_info, "calculate_correlation": calculate_correlation, "create_pivot_table": create_pivot_table, "run_sql_query": run_sql_query, "get_sentiment": get_sentiment, "scrape_wikipedia_summary": scrape_wikipedia_summary, "scrape_pdf_tables": scrape_pdf_tables, "analyze_image_content": analyze_image_content, "geocode_address": geocode_address, "scrape_dynamic_site": scrape_dynamic_site, "parse_html": parse_html, "get_bbc_weather": get_bbc_weather, }
            planner_system_prompt =  f"""
        You are an expert-level data analysis planner. Your purpose is to convert a user's request into a step-by-step JSON execution plan.
        You have been provided with the following available tools:
        {", ".join(AVAILABLE_TOOLS.keys())}
        You must decide on the best strategy to fulfill the request. You have two strategies available:
        **Strategy 1: Use a Specialized Tool.**
        If the user's request can be answered directly and completely by a single call to one of the specialized tools (e.g., `get_bbc_weather`, `geocode_address`, `get_sentiment`), you MUST generate a simple, one-step plan that calls that tool. This is your preferred strategy for simple, direct requests. USE THE SAME NAMES GIVEN TO YOU TO CREATE THE TOOL CALLS.
        **Strategy 2: Generate a Single Python Script.**
        If the user's request is complex, requires multiple steps, data manipulation, or cannot be handled by a single specialized tool, you MUST generate a plan containing a SINGLE step that uses the `python_interpreter`. This single step must contain a complete, self-contained Python script that performs all the necessary actions and prints the final JSON output.
        CRITICAL RULES:
        1.  **CHOOSE A STRATEGY:** First, analyze the request. Is it a simple task for a specialized tool, or a complex one requiring a full script?
        2.  **TOOL NAMES:** You MUST use the exact tool names from the provided list. 
        3. **SPECIAL TOOL USAGE - 
        `analyze_image_content`:
        a. ** If you choose to use the `analyze_image_content` tool, you MUST provide both the `image_path` and a `prompt`. The `prompt` argument **MUST** contain all the specific questions the user has asked about the image, extracted from their main request.**
        b. **For `run_sql_query`, the `db_connection_string` must be formatted correctly. For an uploaded file named 'my_data.db', the correct string is `'sqlite:///my_data.db'`.**
        c. **For `parse_html`, the `selectors` argument MUST be a JSON object mapping descriptive names to CSS selector strings. Example: `{{"titles": "h2.article-title", "prices": ".price-tag"}}`.**
        4.  **DATA CLEANING (IMPORTANT):
        a. ** When you load a CSV file using pandas, the column names might have leading/trailing whitespace. Your first step after loading the data MUST be to clean the column names. A good method is: `df.columns = df.columns.str.strip()`.**
        b. **When identifying columns, you MUST perform a case-insensitive match. A robust method is to convert all column names to lowercase for comparison, like this: `df_cols_lower = [c.lower() for c in df.columns]` and then search for your lowercase keywords (e.g., 'sales') in that list.**
        5.  **FINAL OUTPUT (CRITICAL):** You MUST read the user's request very carefully to determine the exact final output format.
            - If the user asks for a **JSON object with specific keys**, your script's final print statement MUST produce a JSON object with EXACTLY those keys and data types.
            - If the user asks for a **JSON array**, your script's final print statement MUST produce a JSON array with the raw values in the correct order.
        6. **NO PLACEHOLDERS:** You MUST perform the actual calculations and data analysis required. Do not use placeholder or example values in your final output. The results must be derived from the provided data sources.
        7.  Your entire output MUST be ONLY a valid JSON object representing the execution plan. The plan should follow this schema: {{"plan": {{"steps": [{{...}}]}}}}
        """
        
            
            # On retries (i > 0), add the context of the last failure
            if i > 0:
                user_prompt = f"The previous attempt failed.\nPREVIOUS PLAN:\n{json.dumps(last_plan, indent=2)}\n\nPREVIOUS ERROR/OUTPUT:\n{last_error}\n\nPlease analyze the error and generate a new, corrected plan to fulfill the original request:\n{task_content}"
            else:
                user_prompt = f"--- USER REQUEST ---\n{task_content}\n\n--- AVAILABLE FILES ---\n{attached_file_names}"
            
            print("--- Calling Planner LLM to create execution plan ---")
            planner_messages = [{"role": "system", "content": planner_system_prompt}, {"role": "user", "content": user_prompt}]
            
            try:
                response = client.chat.completions.create(model="openai/gpt-5-nano", messages=planner_messages, response_format={"type": "json_object"})
                plan_str = response.choices[0].message.content
                plan = json.loads(plan_str)
                print("\n\n--- 🕵️ DECODING: PLAN RECEIVED 🕵️ ---")
                print(json.dumps(plan, indent=2))
                print("-----------------------------------------\n")

                last_plan = plan
                print("--- Plan received from Planner ---")
            except Exception as e:
                last_error = f"Planner failed to generate a valid JSON plan: {e}"
                if i < max_retries - 1: continue # Go to the next retry attempt
                else: break # Exit loop if retries are exhausted

            # --- 3. EXECUTION ---
            print("--- Starting Worker execution ---")
            final_result = None
            try:
                plan_steps = plan.get("plan", {}).get("steps", [])
                if not plan_steps: raise ValueError("The generated plan contains no steps.")

                for step_data in plan_steps:
                    # Use the final, robust worker logic from our last iteration
                    tool_name = step_data.get("tool_name", step_data.get("tool", step_data.get("action", step_data.get("name"))))
                    if not tool_name: raise ValueError("Plan step is missing a 'tool' or 'action' key.")
                    
                    tool_function = AVAILABLE_TOOLS.get(tool_name)
                    if not tool_function: raise ValueError(f"Plan requested an unknown tool: '{tool_name}'")
                    
                    known_keys = ["step", "id", "tool", "tool_name", "action", "description", "notes", "output"]
                    arguments = {k: v for k, v in step_data.items() if k not in known_keys}
                    if "script" in arguments and "code" not in arguments: arguments["code"] = arguments.pop("script")

                    # Add special context arguments
                    if tool_name in ["python_interpreter", "get_dataframe_info", "calculate_correlation", "create_pivot_table", "scrape_pdf_tables", "analyze_image_content", "scrape_dynamic_site", "parse_html"]: arguments["work_dir"] = work_dir
                    if tool_name in ["get_sentiment", "analyze_image_content"]: arguments["client"] = client

                    # Execute the tool
                    if asyncio.iscoroutinefunction(tool_function): output = await tool_function(**arguments)
                    else: output = tool_function(**arguments)
                    final_result = output
                
                # --- 4. VALIDATION ---
                if is_output_valid(final_result):
                    print("--- Output is valid. Task complete. ---")
                    print("\n\n--- ✅ DECODING: FINAL VALID OUTPUT ✅ ---")
                    print(final_result)
                    print("------------------------------------------\n")

                    return json.loads(final_result)
                else:
                    print("--- Output is invalid. Triggering self-correction. ---")
                    
                    last_error = f"The script executed but produced an invalid result: {final_result}"
            
            except Exception as e:
                print(f"--- Execution failed. Triggering self-correction. ---")
                last_error = f"The worker failed to execute the plan. Error: {repr(e)}"
        
        # If all retries fail, raise the final error
        print(f"--- AGENT FAILED: All {max_retries} attempts exhausted. Returning empty JSON. ---")
        print(f"Last known error was: {last_error}")
        
        # We need to figure out the expected format (list or dict) from the question.
        # A simple heuristic: if the question asks for a JSON object, return {}.
        # If it asks for a JSON array, return [].
        if "JSON object" in task_content:
            return {}
        elif "JSON array" in task_content:
            return []
        else:
            # A safe default if the format is not specified
            return {}

