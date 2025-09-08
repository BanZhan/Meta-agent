import pdfplumber
import csv
import os
import json
import time
import re
import glob
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
from DS_sili_api_3 import deepseek_api_call
from DS_api import deepseek_api
from all_AI_API_3 import all_AI_api

import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

global model_name
'''
claude-3-7-sonnet-20250219-svip
claude-sonnet-4-20250514-svip

claude-sonnet-4-20250514
claude-3-7-sonnet-20250219
claude-3-5-sonnet-20241022

deepseek-r1
deepseek-v3

gemini-2.5-pro-exp-03-25
gemini-2.0-flash-exp

grok-3
grok-2-1212
grok-beta

gpt-4o
gpt-4o-mini
gpt-3.5-turbo
gpt-4-turbo
'''
model_name = 'claude-3-7-sonnet-20250219'

target_extracted_info = """Sampling_Region | Sampling_Year | Sampling_Month | Sampling_Coordinate_lat | Sample_Coordinate_lon | Sampling_Method | Sampling_Depth | Mean_MP_Density/Abundance/Concentration | MP_Morphology | Mean_MP_Size | MP_Color | MP_Polymer"""
current_task_sentence = """focusing on microplastics sampling in the global ocean"""
search_term = """(microplastic* OR "micro plastic*" OR "micro-plastic*" OR microlitter* OR "micro-litter*" OR microparticle* OR micro-particle* OR "micro debris" OR "plastic particle*" OR "nano-plastic*" OR "nano plastic*" OR "synthetic fiber*" OR "textile fiber*") AND
(concentrat* OR quantif* OR amount* OR abundance OR measure* OR level* OR density OR distribut* OR "particle weight") AND
(marine OR ocean* OR sea OR coast* OR estuary OR bay OR sediment* OR Pacific OR Atlantic OR Arctic OR seawater)"""

# Improved prompt template with clearer instructions and expected output format
Sentence_Extraction_PROMPT = """
Please as a scientist on meta-analysis, extract and output original sentences and full tables with related information and data according to the information:
current task: {current_task_sentence};
target_extracted_information: {target_extracted_info};
search_term: {search_term};
from the following input text.

Constraints:
Do extract the direct data and information from the paper.
Do not paraphrase, modify, or interpret any content of original sentences or tables.
Do not omit any content of tables.
The input text:
"""
Sentence_Extraction_PROMPT = Sentence_Extraction_PROMPT.format(current_task_sentence=current_task_sentence, target_extracted_info=target_extracted_info, search_term=search_term)


Whole_Text_Classification_PROMPT = """Pretend to be a scientist. Does the following input text contain related data, according to
current task: {current_task_sentence};
target_extracted_information: {target_extracted_info};
search_term: {search_term};
Answer 'Yes' or 'No' only. Do not explain. No explanation! No reasoning!
Input text:
"""
Whole_Text_Classification_PROMPT = Whole_Text_Classification_PROMPT.format(current_task_sentence=current_task_sentence, target_extracted_info=target_extracted_info, search_term=search_term)


######## factor and data extraction
Value_extraction_PROMPT = """Task: Please as a scientist on meta-analysis, extract and return data from the input text and tables about the information:
Columns (order matters): {target_extracted_info};

Follow ALL rules:
Extract all related results correctly in the present study.
Separate columns with |. Use \n for line breaks.
If data is missing, replace the field with None. Do NOT leave fields blank or invent other placeholders.
Preserve exact wording, units, and formatting from the text (e.g., "pH: 7.2", "lead: 0.015 mg/L").
One row for one sample.
Return only the structured data. No headers, explanations, or markdown.

Input Text:
"""
Value_extraction_PROMPT = Value_extraction_PROMPT.format(target_extracted_info=target_extracted_info)


Check_for_extracted_values_PROMPT = """ Double-check the extracted data. Please be very strict.
Columns (order matters): {target_extracted_info};
Extracted data 1: {extracted_data_1};
Extracted data 2: {extracted_data_2};
Extracted data 3: {extracted_data_3};
Please screen the extracted data for accuracy, high quality and validity of key results.
Follow ALL rules:
One row of data for one sample.
Separate columns with |. Use \n for line breaks.
Return only the structured data. No headers, explanations, or markdown.
"""


#Clean_extracted_results_PROMPT = """As a scientist on microplastics, please clean the extracted results in each column to improve the quality of the extracted data; 
#then, return the cleaned data in an JSON format"""

def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from API response text."""
    try:
        # Look for JSON pattern in response
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        else:
            # Try to load the entire response as JSON
            return json.loads(response)
    except json.JSONDecodeError:
        logger.error(f"Could not extract valid JSON from response")
        return None

def extract_text_from_pdf_page(pdf, page_num: int) -> str:
    """Extract text from a specific page of a PDF."""
    try:
        page = pdf.pages[page_num]
        return page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error extracting text from page {page_num}: {str(e)}")
        return ""

def extract_tables_from_pdf_page(pdf, page_num: int) -> list:
    """Extract tables from a specific page of a PDF."""
    try:
        page = pdf.pages[page_num]
        tables = page.extract_tables()
        return tables or []
    except Exception as e:
        logger.error(f"Error extracting tables from page {page_num}: {str(e)}")
        return []

def call_api_with_retries(task_type, prompt: str) -> Dict[str, Any]:
    """Call the API with retries on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"API call attempt {attempt + 1}/{MAX_RETRIES}")
            #response = deepseek_api_call(prompt)
            if task_type == 'extract_original_sentences':
                response = deepseek_api_call(prompt)
                #response = deepseek_api('deepseek-chat', prompt)
                #response = all_AI_api('deepseek-v3', prompt)
            elif task_type == 'value_extraction':
                #response = deepseek_api_call(prompt)
                #response = deepseek_api('deepseek-chat', prompt)
                response = all_AI_api(model_name, prompt)
            else:
                #response = deepseek_api_call(prompt)
                #response = deepseek_api('deepseek-chat', prompt)
                response = all_AI_api(model_name, prompt)

            if response:
                response = response.replace('"', '').replace("'", "").strip().lower()
                return response
            
            logger.warning(f"API returned unexpected format: {response[:200]}...")
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            time.sleep(RETRY_DELAY * (attempt + 1))  # Progressive backoff
            
    
    logger.error("Maximum retries reached, returning empty result")
    return "None results"


def save_results(data: Dict[str, Any], output_path: str) -> None:
    """Save extracted data to JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")


def save_intermediate_results(data: Dict[str, Any], output_dir: str = "intermediate_results") -> None:
    """Save intermediate results during processing."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"output.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Intermediate results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving intermediate results: {str(e)}")


def append_to_csv(filepath, data, file_path='extracted_data_save.csv'):
    # Define column names based on the data structure
    columns = [
        'Sampling_Region', 'Sampling_Year', 
        'Sampling_Month', 'Sampling_Coordinate_lat', 'Sample_Coordinate_lon', 'Sampling_Method', 
        'Sampling_Depth', 'Mean_MP_Density/Abundance/Concentration', 'MP_Morphology', 
        'Mean_MP_Size', 'MP_Color', 'MP_Polymer'
    ]

    # Split the data into rows
    rows = [line.split('|') for line in data.strip().split('\n')]
    
    # Validate that each row has the correct number of fields
    valid_rows = [row for row in rows if len(row) == len(columns)]
    
    if not valid_rows:
        print("Warning: No valid rows found in the data.")
        return

    # Convert rows to a list of dictionaries
    data_dicts = [dict(zip(columns, row)) for row in valid_rows]

    # Create DataFrame
    df = pd.DataFrame(data_dicts)
    df['filepath'] = filepath

    # Append to CSV file (create if it doesn't exist, append if it does)
    try:
        # Check if file exists to determine whether to write headers
        file_exists = os.path.isfile(file_path)
        
        # Append mode, write headers only if file doesn't exist
        df.to_csv(file_path, mode='a', index=False, header=not file_exists)
        
        print(f"Data successfully appended to {file_path}")
        print("First few rows of the DataFrame:")
        print(df.head())
        
    except Exception as e:
        print(f"Error writing to CSV: {e}")


def main():
    # Load environment variables
    load_dotenv()
    # File paths
    # Define directories
    pdf_dir = "data_extraction_examples_20_pdfs"
    output_path = "extracted_data_0617_23_pdfs_claude35.json"
    # Get list of PDF files
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    #print(pdf_files)

    # Process each PDF file
    i = 0
    for pdf_path in pdf_files[i:]:
        # Step 1: Open the PDF
        logger.info(f"Opening PDF file: {i+1, pdf_path}")
        i = i+1
        # Generate output path
        pdf_filename = os.path.basename(pdf_path)
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                
                logger.info(f"PDF has {page_count} pages")
                
                all_pages_data = ''  # Collect page' text and tables
                
                for page_num in range(0, page_count):

                    page_text = extract_text_from_pdf_page(pdf, page_num)
                    page_table = extract_tables_from_pdf_page(pdf, page_num)

                    # Collect data for each page
                    page_data = f"\n\n--- PAGE {page_num+1} ---\n\n{page_text}  ---\n\n{page_table} "
                    #print('page_data', page_data)
                    # Sentence extraction prompt with extracted text for each page
                    # Format prompt with extracted text
                    sentence_prompt = Sentence_Extraction_PROMPT + page_data
                    #print('sentence prompt: ', sentence_prompt)
                    #sentence_prompt = Sentence_Extraction_PROMPT.format(input_text=page_data, extracted_info=all_pages_data)
                    # Call API to extract structured data 
                    #print('page number: ', page_num)
                    batch_result = call_api_with_retries('extract_original_sentences', sentence_prompt) # extract_original_sentences
                    #print('batch_result: ', batch_result)
                    #print('extracted related sentences: ', batch_result)
                    all_pages_data = all_pages_data + '[PAGE number: ' + str(page_num+1) + ']' + batch_result
                    
                    # Add a small delay between batches to avoid API rate limits
                    time.sleep(0.1)

                    #break

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return

        #break

        final_result = all_pages_data

        # transfer the json savings into a txt with csv structures
        #json_to_DFtxt()

        # Format prompt with extracted text
        text_classification_prompt = Whole_Text_Classification_PROMPT + all_pages_data

        whole_text_try = 0; classification_yes = 0
        while whole_text_try < 5 and classification_yes < 1:
            # Call API to classify the whole text data
            text_classification_result = call_api_with_retries('whole_text_classification', text_classification_prompt)
            if text_classification_result == 'yes':
                classification_yes += 1
            
            whole_text_try += 1

        print('classification yes: ', classification_yes)
        if classification_yes >= 1:

            #### Call API to extract data points from texts
            text_data_extraction_result_1 = call_api_with_retries('value_extraction', Value_extraction_PROMPT + all_pages_data)
            text_data_extraction_result_2 = call_api_with_retries('value_extraction', Value_extraction_PROMPT + all_pages_data)
            text_data_extraction_result_3 = call_api_with_retries('value_extraction', Value_extraction_PROMPT + all_pages_data)
            print('text_data_extraction_result_1: ', text_data_extraction_result_1)
            print('text_data_extraction_result_2: ', text_data_extraction_result_2)
            print('text_data_extraction_result_3: ', text_data_extraction_result_3)
            final_check_PROMPT = Check_for_extracted_values_PROMPT.format(target_extracted_info=target_extracted_info, extracted_data_1=text_data_extraction_result_1, extracted_data_2=text_data_extraction_result_2, extracted_data_3=text_data_extraction_result_3)

            check_for_extracted_values = call_api_with_retries('others', final_check_PROMPT)

            if not check_for_extracted_values:
                print('No valid information for the paper.')
                continue
            
        else:
            continue

        append_to_csv(pdf_path, check_for_extracted_values, file_path='extracted_data.csv')

        # After processing all pages, save to save_all.json
        pdf_data = {
            "file_no.": i,
            "pdf_path": pdf_path,
            "extracted_data": check_for_extracted_values
            } #"extracted_info": all_pages_data,

        # Load existing data and append new entry
        save_all_path = "save_all.json"
        if os.path.exists(save_all_path):
            with open(save_all_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        else:
             existing_data = []
        
        existing_data.append(pdf_data)
        
        with open(save_all_path, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()