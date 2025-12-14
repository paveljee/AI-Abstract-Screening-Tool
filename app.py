#Copyright (C) 2024 [Jin Kyu Kim, MD]. All Rights Reserved.
#Article Screening Tool
#Copyright (C) 2024 [Jin Kyu Kim, MD]
#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version. 
#This program comes with ABSOLUTELY NO WARRANTY; for details refer to LICENSE file. This is free software, and you are welcome to redistribute it under certain conditions; refer to LICENSE file for details.
#Additional restrictions apply: No commercial use without permission, Attribution required, Network use restrictions apply
#Contact: [jjk.kim@mail.utoronto.ca]

# Flask and other imports
from flask import Flask, render_template, request, Response, session, stream_with_context, send_file
import os
import pandas as pd
import openai
import asyncio
import aiohttp
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import logging
import json
import traceback
import time
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from flask_session import Session
import csv
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'jinkyukimsafary123qwoierjqwrevnx'  # Replace with a random string
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'ris'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['SESSION_TYPE'] = 'filesystem'

Session(app)

# Debug environment variable loading
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    logging.error("OpenAI API key is not set in environment variables.")
else:
    logging.info("OpenAI API key successfully loaded from environment variables.")

openai.api_key = openai_api_key

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def read_ris_files(file_paths):
    all_entries = []
    total_entries = 0
    title_fields = ['TI', 'T1', 'CT', 'BT']  # Common RIS tags for titles
    abstract_fields = ['AB', 'N2']  # Common RIS tags for abstracts
    author_fields = ['AU', 'A1', 'A2', 'A3', 'A4']  # RIS tags for authors
    year_fields = ['PY', 'Y1']  # RIS tags for publication year

    for file_path in file_paths:
        try:
            logging.info(f"Starting to read file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                current_entry = {}
                for line in file:
                    line = line.strip()
                    if line.startswith('TY  - '):
                        if current_entry:
                            total_entries += 1
                            process_entry(current_entry, all_entries, title_fields, abstract_fields, author_fields, year_fields)
                        current_entry = {'type': line[6:]}
                    elif ' - ' in line:
                        tag, content = line.split(' - ', 1)
                        tag = tag.strip()
                        content = content.strip()
                        if tag in current_entry:
                            if isinstance(current_entry[tag], list):
                                current_entry[tag].append(content)
                            else:
                                current_entry[tag] = [current_entry[tag], content]
                        else:
                            current_entry[tag] = content
                    elif line == 'ER  -':
                        if current_entry:
                            total_entries += 1
                            process_entry(current_entry, all_entries, title_fields, abstract_fields, author_fields, year_fields)
                        current_entry = {}

                # Process the last entry if file doesn't end with ER  -
                if current_entry:
                    total_entries += 1
                    process_entry(current_entry, all_entries, title_fields, abstract_fields, author_fields, year_fields)

            logging.info(f"Successfully read {total_entries} entries from {file_path}")
        except Exception as e:
            logging.error(f"Error reading {file_path}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")

    logging.info(f"Total entries read from all files: {total_entries}")
    logging.info(f"Entries with valid titles: {len(all_entries)}")
    return all_entries

def process_entry(entry, all_entries, title_fields, abstract_fields, author_fields, year_fields):
    title = ''
    for field in title_fields:
        if field in entry:
            title = entry[field]
            if isinstance(title, list):
                title = ' '.join(title)
            title = title.strip()
            if title:
                break

    abstract = ''
    for field in abstract_fields:
        if field in entry:
            abstract = entry[field]
            if isinstance(abstract, list):
                abstract = ' '.join(abstract)
            abstract = abstract.strip()
            if abstract:
                break

    first_author = ''
    for field in author_fields:
        if field in entry:
            authors = entry[field]
            if isinstance(authors, list):
                first_author = authors[0].strip()
            else:
                first_author = authors.strip()
            break

    year = ''
    for field in year_fields:
        if field in entry:
            year = entry[field].strip()
            break

    if title:
        all_entries.append({
            'title': title,
            'abstract': abstract,
            'first_author': first_author,
            'year': year
        })
        logging.debug(f"Added entry: Title: {title[:50]}...")
    else:
        logging.warning(f"Skipping entry without title: {entry}")

def remove_duplicates(entries):
    if not entries:
        raise ValueError("No entries found in the data")
    
    df = pd.DataFrame(entries)
    logging.debug(f"DataFrame before deduplication:\n{df}")
    
    if 'title' not in df.columns:
        raise ValueError("No 'title' column found in the data")
    
    df['title'] = df['title'].astype(str)
    df.drop_duplicates(subset=['title'], inplace=True)
    logging.info(f"Removed duplicates. {len(df)} unique entries remaining.")
    
    logging.debug(f"DataFrame after deduplication:\n{df}")
    return df.to_dict('records')

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def screen_article_async(session: ClientSession, title: str, abstract: str, criteria: str):
    prompt = f"""
    As an expert in systematic reviews, your task is to determine if the following article should be included in a systematic review based on its title and abstract (if available).

    Title: {title}
    Abstract: {abstract if abstract else "Not available"}

    Inclusion criteria:
    {criteria}

    Please respond with 'Include' if the article meets the criteria, 'Exclude' if it does not, or 'Unclear' if you cannot make a determination. Then, provide a brief explanation for your decision.

    Response format:
    Decision: [Include/Exclude/Unclear]
    Explanation: [Your reasoning here]
    """

    try:
        logging.debug(f"Sending request to OpenAI API for title: {title[:50]}...")
        use_llama_cpp = 'llama' in str(openai.api_key).lower()
        async with session.post(
            'https://api.openai.com' if not use_llama_cpp else 'http://localhost:8000' + '/v1/chat/completions',
            json={
                "model": "gpt-4o-mini" if not use_llama_cpp else "ggml-org/gpt-oss-20b-GGUF",
                "messages": [
                    {"role": "system", "content": "You are an expert in systematic reviews, tasked with screening articles for inclusion."},
                    {"role": "user", "content": prompt}
                ]
            },
            headers={
                "Authorization": f"Bearer {openai.api_key}"
            } if not use_llama_cpp else None,
            timeout=ClientTimeout(total=60)
        ) as response:
            response.raise_for_status()
            result = await response.json()
            logging.debug(f"API response: {json.dumps(result, indent=2)}")

            if 'choices' in result and result['choices'] and 'message' in result['choices'][0]:
                content = result['choices'][0]['message']['content']
                
                decision = None
                explanation = None
                for line in content.split('\n'):
                    if line.lower().startswith('decision:'):
                        decision = line.split(':', 1)[1].strip()
                    elif line.lower().startswith('explanation:'):
                        explanation = line.split(':', 1)[1].strip()
                
                if decision and explanation:
                    return f"Decision: {decision}\nExplanation: {explanation}"
                else:
                    logging.error(f"Unable to parse decision and explanation from response: {content}")
                    return None
            else:
                logging.error(f"Unexpected API response structure: {json.dumps(result, indent=2)}")
                return None

    except Exception as e:
        logging.error(f"Error in API call: {e}")
        logging.error(traceback.format_exc())
        raise

async def process_files_async(entries, criteria, num_duplicates):
    logging.info(f"Starting to process {len(entries)} entries")
    
    start_time = time.time()
    connector = TCPConnector(limit=5)
    
    async with ClientSession(connector=connector) as session:
        screened_entries = []
        num_included = 0
        num_excluded = 0
        num_unclear = 0

        for i, entry in enumerate(entries):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = await screen_article_async(session, entry['title'], entry['abstract'], criteria)
                    elapsed_time = time.time() - start_time
                    progress = (i + 1) / len(entries) * 100
                    remaining_time = (elapsed_time / (i + 1)) * (len(entries) - (i + 1))

                    yield {
                        'progress': progress,
                        'elapsed_time': elapsed_time,
                        'remaining_time': remaining_time,
                        'processed': i + 1,
                        'total': len(entries)
                    }

                    if result:
                        decision_line = result.split('\n')[0]
                        explanation_line = result.split('\n')[1]
                        
                        decision = decision_line.split(': ')[1].lower()
                        explanation = explanation_line.split(': ')[1]
                        
                        if decision == 'include':
                            entry['include'] = True
                            num_included += 1
                        elif decision == 'exclude':
                            entry['include'] = False
                            num_excluded += 1
                        else:
                            entry['include'] = 'unclear'
                            num_unclear += 1
                        
                        entry['explanation'] = explanation
                        break  # Success, exit retry loop
                    else:
                        raise ValueError("No valid result returned")
                
                except Exception as e:
                    logging.error(f"Error processing entry (attempt {attempt + 1}): {str(e)}")
                    if attempt == max_retries - 1:  # Last attempt
                        entry['include'] = 'unclear'
                        entry['explanation'] = f"Error: Unable to process after {max_retries} attempts. Last error: {str(e)}"
                        num_unclear += 1
            
            screened_entries.append(entry)
            logging.info(f"Processed {len(screened_entries)} out of {len(entries)} entries")
            
            await asyncio.sleep(0.1)  # Small delay to prevent overwhelming the API
    
    total_time = time.time() - start_time
    logging.info(f"Screened {len(screened_entries)} entries in {total_time:.2f} seconds.")
    logging.info(f"Included: {num_included}, Excluded: {num_excluded}, Unclear: {num_unclear}")
    
    summary = {
        'total_entries': len(entries) + num_duplicates,
        'duplicates_removed': num_duplicates,
        'screened': len(screened_entries),
        'included': num_included,
        'excluded': num_excluded,
        'unclear': num_unclear
    }
    
    csv_filename = 'screened_articles.csv'
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Title', 'Abstract', 'First Author', 'Year', 'Decision', 'Explanation'])
        for entry in screened_entries:
            decision = 'Include' if entry['include'] is True else ('Exclude' if entry['include'] is False else 'Unclear')
            csvwriter.writerow([
                entry['title'],
                entry['abstract'],
                entry['first_author'],
                entry['year'],
                decision,
                entry['explanation']
            ])
    
    yield {
        'progress': 100,
        'elapsed_time': total_time,
        'remaining_time': 0,
        'processed': len(entries),
        'total': len(entries),
        'summary': summary,
        'csv_filename': csv_filename
    }

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'Error: No file part', 400
    
    files = request.files.getlist('file')
    criteria = request.form['criteria']
    
    if not files or files[0].filename == '':
        return 'Error: No selected file', 400
    
    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            file_paths.append(file_path)
    
    if not file_paths:
        return 'Error: No valid files uploaded', 400
    
    session['file_paths'] = file_paths
    session['criteria'] = criteria
    
    return 'Files uploaded successfully', 200

@app.route('/process')
def process_files():
    file_paths = session.get('file_paths', [])
    criteria = session.get('criteria', '')
    
    if not file_paths:
        return 'Error: No files to process', 400
    
    entries = read_ris_files(file_paths)
    if not entries:
        return 'Error: No valid entries found in the uploaded files', 400
    
    original_count = len(entries)
    unique_entries = remove_duplicates(entries)
    num_duplicates = original_count - len(unique_entries)
    
    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def async_generator():
            async for status in process_files_async(unique_entries, criteria, num_duplicates):
                yield f"data: {json.dumps(status)}\n\n"

        async def run_async():
            async for item in async_generator():
                yield item

        for item in loop.run_until_complete(collect_results(run_async())):
            yield item

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/download_csv/<filename>')
def download_csv(filename):
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(csv_path, as_attachment=True, download_name=filename)

async def collect_results(async_gen):
    results = []
    async for item in async_gen:
        results.append(item)
    return results

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
