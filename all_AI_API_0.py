import os
import openai


# openai.api_key = 
openai.api_key = ""
openai.base_url = ""
openai.default_headers = {"x-foo": "true"}

def all_AI_api(model_name, content):
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
    try:
        completion = openai.chat.completions.create(
            model=model_name,
            messages=[
                {
                "role": "user",
                "content": content,
                },
            ],
            temperature=0,
            max_tokens = 8000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return None


#### test
#print(all_AI_api('gpt-4o', 'who are you'))
