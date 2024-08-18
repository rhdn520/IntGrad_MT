# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
load_dotenv()
import json
from tqdm import tqdm
import time
from openai import OpenAI
import anthropic

class GPTTranslator:
    def __init__(self, model_id = "gpt-3.5-turbo") -> None:
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_id = model_id

    def simple_translation(self, sent, target_lang="Korean", max_retries = 3, temperature=0.3):
        messages = [
            {
                "role": "system",
                "content": f"You will be provided with a sentence in English, and your task is to translate it into {target_lang}."
            },
            {
                "role":"user",
                "content": sent
            }
        ]
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model= self.model_id,
                    messages= messages,
                    temperature= temperature
                )
                
                return response.choices[0].message.content # Exit retry loop if successful response
                    
            except Exception as e:
                print(f"An error occurred on attempt {retries + 1}: {e}")
                time.sleep(5)
                retries += 1
                if retries >= max_retries:
                    print("Max retries reached. Exiting...")
                    return 'error'

    def grad_translation(self, start, start_mt, end, intp_sents, target_lang="Korean", max_retries = 3, temperature=0.3):
        intp_sents = intp_sents[1:-1]

        messages = [
            {
                "role": "system",
                "content": f"You will be provided with a sentence in English, and your task is to translate it into {target_lang}."
            },
            {
                "role":"user",
                "content": start
            },
            {
                "role":"assistant",
                "content": start_mt
            }
        ]

        for intp_sent in intp_sents:
            retries = 0  # Initialize retries inside the loop
            
            messages.append({
                "role":"user",
                "content":intp_sent
            })

            while retries < max_retries:
                try:
                    response = self.client.chat.completions.create(
                        model= self.model_id,
                        messages= messages,
                        temperature= temperature
                    )
                    messages.append({
                        "role":"assistant",
                        "content": response.choices[0].message.content
                    })
                    break  # Exit retry loop if successful response
                    
                except Exception as e:
                    print(f"An error occurred on attempt {retries + 1}: {e}")
                    time.sleep(5)
                    retries += 1
                    if retries >= max_retries:
                        print("Max retries reached. Exiting...")
                        return 'error', ['error']

                    
        messages.append({
            "role":"user",
            "content":end
        })

        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=temperature
                )
                end_mt = response.choices[0].message.content
                grad_mts = [x["content"] for x in messages if x["role"]=="assistant"]
                return end_mt, grad_mts
                
            except Exception as e:
                print(f"An error occurred on attempt {retries + 1}: {e}")
                time.sleep(5)
                retries += 1

                if retries >= max_retries:
                    print("Max retries reached. Exiting...")
                    return 'error', ['error']

    def aggr_translation(self, src, mts, target_lang="Korean", max_retries=3, temperature=0.3):
        messages = [
            {
                "role": "system",
                "content": f"You will be provided with a sentence in English, and your task is to translate it into {target_lang}."
            }
        ]

        for mt in mts:
            messages.append(
                {
                    "role":"user",
                    "content":src
                }
            )
            messages.append(
                {
                    "role":"assistant",
                    "content":mt
                }
            )
        
        messages.append(
            {
                "role":"user",
                "content":src
            }
        )

        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature
                )
                end_mt = response.choices[0].message.content
                return end_mt
                
            except Exception as e:
                print(f"An error occurred on attempt {retries + 1}: {e}")
                time.sleep(5)
                retries += 1

                if retries >= max_retries:
                    print("Max retries reached. Exiting...")
                    return 'error' 


    def fewshot_translation(self, src, examples, target_lang="Korean", max_retries=3, temperature=0.3):
        """Translation with few shot examples

        Args:
            src (str): 번역할 문장
            examples (list): dictionary들의 list. key로는 'src'와 'ref'
        """

        messages = [
            {
                "role": "system",
                "content": f"You will be provided with a sentence in English, and your task is to translate it into {target_lang}."
            }
        ]

        for example in examples:
            messages.append({
                "role":"user",
                "content": example['src']
            })
            messages.append({
                "role":"assistant",
                "content": example['ref']
            })
        
        messages.append(
            {
                "role":"user",
                "content":src
            }
        )

        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model= self.model_id,
                    messages= messages,
                    temperature= temperature
                )
                
                return response.choices[0].message.content # Exit retry loop if successful response
                    
            except Exception as e:
                print(f"An error occurred on attempt {retries + 1}: {e}")
                time.sleep(5)
                retries += 1
                if retries >= max_retries:
                    print("Max retries reached. Exiting...")
                    return 'error'        

        return 'error'