################################################Prepare Your Environment
import csv
import requests
import os
import re
import time
import json
from dotenv import load_dotenv
from all_AI_API_3 import all_AI_api
import chardet


################################################Create Classification Prompt

target_extracted_info = """Paper_Title | Publish_Year | Publish_Journal | Sampling_Region | Sampling_Year | Sampling_Month | Sampling_Coordinate_lat | Sample_Coordinate_lon | Sampling_Method | Sampling_Depth | Mean_MP_Density/Abundance/Concentration | MP_Morphology | Mean_MP_Size | MP_Color | MP_Polymer"""
current_task_sentence = """focusing on microplastics sampling in the global ocean"""
search_term = """(microplastic* OR "micro plastic*" OR "micro-plastic*" OR microlitter* OR "micro-litter*" OR microparticle* OR micro-particle* OR "micro debris" OR "plastic particle*" OR "nano-plastic*" OR "nano plastic*" OR "synthetic fiber*" OR "textile fiber*") AND
(concentrat* OR quantif* OR amount* OR abundance OR measure* OR level* OR density OR distribut* OR "particle weight") AND
(marine OR ocean* OR sea OR coast* OR estuary OR bay OR sediment* OR Pacific OR Atlantic OR Arctic OR seawater)"""


CLASSIFICATION_PROMPT = """
current task: {current_task_sentence};
target_extracted_information: {target_extracted_info};
search_term: {search_term};
Please as a scientist on meta-analysis, classify whether the following scientific paper contains related data and information according to the above information:
Title: {title}
Abstract: {abstract}
Answer 'Yes' or 'No' only..Do not explain! No explanation! No reasoning!
"""


# Load API key from .env
load_dotenv()

# Initialize API client
# 修改后的分类函数
def classify_paper(title, abstract):
    prompt = CLASSIFICATION_PROMPT.format(current_task_sentence=current_task_sentence, target_extracted_info=target_extracted_info, search_term=search_term, title=title, abstract=abstract)
    try_t = 0
    while try_t < 5:
        # 获取原始响应
        response_str = all_AI_api(model_name, prompt)
        print(f"[DEBUG] 原始响应: {response_str}")  # 添加调试输出

        # 尝试解析JSON
        try:
            response_dict = json.loads(response_str)
            result = response_dict['choices'][0]['message']['content']
        except json.JSONDecodeError:
            # 如果直接返回文本内容
            result = response_str.strip()
        
        # 统一清理结果
        clean_result = result.replace('"', '').replace("'", "").strip().lower()


        print(f"[DEBUG] 清理后结果: {clean_result}")  # 调试输出
        
        if clean_result == 'yes':
            return 1
        elif clean_result == 'no':
            return 0
        elif 'yes' in clean_result:
            return 1
        else:
            try_t += 1

    #print(f"处理异常: {str(e)}")
    return 0  # 默认返回不符合条件


################################################Process CSV File
def process_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        new_columns = reader.fieldnames + ['pred_positive']
        
        # 主输出文件的写入器
        writer = csv.DictWriter(outfile, fieldnames=new_columns)
        writer.writeheader()

        # 备份文件设置
        backup_file = 'save.csv'
        file_exists = os.path.isfile(backup_file)
        file_size = os.path.getsize(backup_file) if file_exists else 0

        # 以追加模式打开备份文件
        with open(backup_file, 'a', newline='', encoding='utf-8') as savefile:
            save_writer = csv.DictWriter(savefile, fieldnames=new_columns)
            
            # 如果备份文件为空，写入标题行
            if file_size == 0:
                save_writer.writeheader()

            for row in reader:
                # 分类处理
                classification = classify_paper(row['title'], row['abstract'])
                print(f"Processing: {row['title'][:30]}... | Result: {classification}")
                
                # 设置预测结果，处理非法值
                row['pred_positive'] = classification if classification in (0, 1) else -1
                
                # 写入主文件
                writer.writerow(row)
                # 实时写入备份文件
                save_writer.writerow(row)
                
                # 速率控制
                time.sleep(0.1)
                #break


################################################Run the Classification
def training_testing_data():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	input_file = os.path.join(script_dir, 'performance_evaluation', 'standard_literature_dataset.csv')
	output_file = os.path.join(script_dir, 'performance_evaluation', 'title_abstract_examples_output_' + model_name + '_0618_' +str(repeats) + '.csv')
	process_csv(input_file, output_file)

def paper_classification():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	input_file = os.path.join(script_dir, 'paper_classify_title_abstract', 'combined_data_wos_Springer_deduplicate0501.csv') # test.csv
	output_file = os.path.join(script_dir, 'paper_classify_title_abstract', 'combined_data_wos_Springer_deduplicate0501_0528_claude_3_7_'+ str(repeats) +'.csv')
	process_csv(input_file, output_file)

#############################################################
'''
    {
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
    }
    '''


global model_name, repeats


#'gemini-2.0-flash-exp', 'gemini-2.5-pro-exp-03-25', 'gpt-3.5-turbo', 'gpt-4-turbo', 
# 'grok-beta',
'''
'claude-3-7-sonnet-20250219-svip', 
'claude-sonnet-4-20250514-svip',  
'claude-3-5-sonnet-20241022-svip',
'deepseek-r1',
'deepseek-v3',
'grok-3',
'grok-2-1212',
'''
models = [
'gpt-4o',
'gpt-4o-mini',
'gpt-4-turbo',
'gpt-3.5-turbo'
]

for model_name in models:
    for repeats in range(1, 3):
        print(model_name, '_', repeats)
        training_testing_data()
        #break

'''
model_name = 'claude-3-7-sonnet-20250219-svip'
for repeats in range(1, 4):
    paper_classification() # V3_pro'''