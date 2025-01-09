
from enum import Enum
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
import time
import pandas as pd




client = OpenAI(api_key='api_key', organization='organization', project='project')
 
prompt_template = '''You are an expert news annotator. Your job is to classify the TEXT given belo
w into predefined topic categories strictly.
The topic categories available to you are given in the FORMAT INSTRUCTIONS
###FORMAT INSTRUCTIONS
{format_instructions}
Do not start the JSON with 'Here is the output in JSON format'. Just output the JSON and nothing else
###TEXT
{input}
###QUESTION:
Which predefined topics does the text above talk about? Choose only from what is provided above in the format instructions. If you do not know answer 'unknown'
'''
class Topic(Enum):
    Bible = "Bible"
    Sin = "Sin"
    
from typing import List
# Class to receive and validate user input
class TopicSelection(BaseModel):
    topic: List[Topic]
        
        
def update_dataframe_with_retry(df, row, index):
    parser =  PydanticOutputParser(pydantic_object=TopicSelection)
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    max_attempts = 5
    attempt = 0
    while attempt < max_attempts:
        try:
            print("*******************************\n")
            temp_text = {'input': row['translated_combined_text']}
            #print(temp_text)
            #temp_text = "video_anxiety-30sec_fr\\nDo you feel like anxiety is constantly pulling you in all directions?   You are far from being alone. But there is hope. Stop letting anxiety control your life, but instead discover the peace and love of Jesus for you today. 💬 Click the link to find that peace today.\\nDo you feel like anxiety is constantly pulling you in all directions?   You are far from being alone. But there is hope. Stop letting anxiety control your life, but instead discover the peace and love of Jesus for you today. 💬 Click the link to find this peace today. Can\'t stand being overwhelmed by your anxiety anymore?   Jesus understands your struggles and wants to offer you a way out.   💬 Click on the link to discover the peace and love that Jesus wants to bring you. Anxiety makes life feel like a never-ending cycle of fears and worries.   But there is hope. Discover how Jesus wants to bring you peace and stability.  💬 Click the link to experience this peace today. Anxiety doesn\'t have to leave you feeling helpless and alone... Discover how Jesus can calm your fears and support you through your life\'s challenges.   💬 Click on the link to discover the peace of Jesus today. Do you feel like anxiety is dominating your life?   There is a peace that can calm your fears.    💬 Click the link to find peace today.\\nCome talk to us 2023-12-12-02acab1f83adb36357af0c219157d9e3\\nCome talk to us Message us Let\'s talk today Talk to someone Start a conversation\\n\\nCome talk to us 2023-12-12-02acab1f83adb36357af0c219157d9e3"
            start_time = time.time()  # Start time profiling
            #output = chain.invoke(temp_text)  # Attempt the risky operation
            text_2 = prompt.format(input=temp_text)
            print(text_2)
            print("\n")
            #output = client.completions.create(model="Hermes-2-Pro-Mistral-7B", prompt=text_2, temperature=0.1, max_tokens=1000,input=None, output=None,top_k=50,top_p=0.99)
            output =  client.chat.completions.create(    
                                                        model= "gpt-4",   
                                                        messages=[
                                                            {"role": "system", "content": "You are a topic assigner with critical reasoning"},
                                                            {"role": "user", "content": text_2}
                                                        ],
                                                        temperature=0.1
                                                    )
            end_time = time.time()  # End time profiling
            duration = end_time - start_time  # Calculate the duration
            print(f"Operation time: {duration} seconds")  # Print the duration
            aoe_analysis_data = parse_aoe_analysis(output)
            print(aoe_analysis_data)
            # Update the DataFrame with new columns for each key in the AOEAxnalysis data
            for key, value in aoe_analysis_data.items():
                if isinstance(value, list):  # Check if the value is a list
                    value = ', '.join(value)  # Convert list to comma-separated string
                df.at[index, 'topic'] = value
            print(f"Successfully updated row at index {index}")
            return  # Operation succeeded, exit the function
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            end_time = time.time()  # End time profiling
            duration = end_time - start_time  # Calculate the duration
            print(f"Operation time: {duration} seconds")  # Print the duration
            attempt += 1  # Increment the attempt counter
            time.sleep(1)  # Optional: sleep for a second before retrying
            if attempt == max_attempts:
                print(f"Max attempts reached for index {index}, moving to next row.")
    return df
import json
def parse_aoe_analysis(output):
    print(output)
    """Parse the AOEAnalysis object to a dictionary."""
    # Convert the AOEAnalysis part of the output to a dictionary
    aoe_analysis_json = json.loads(extract_json(output.choices[0].message.content))
    return aoe_analysis_json
 
import re
def extract_json(data):
    # This regex matches the JSON object starting with '{' and ending with '}'
    match = re.search(r'\{[^{}]*\}', data)
    if match:
        return match.group()
    else:
        return "No JSON found"
count = 0
for index, row in df.iterrows():
    count = count + 1
    print("Count: %d", count)
    print(f"Processing index {index}")
    update_dataframe_with_retry(df,row, index)
    progress = (index + 1) / len(df) * 100
    
df['topic'] = df['topic'].str.split(',')  # Split the 'topic' column by comma
df_exploded = df.explode('topic')
df_exploded

