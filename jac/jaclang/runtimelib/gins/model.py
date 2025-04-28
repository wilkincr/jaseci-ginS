from typing import List, TypedDict
import enum
import json

#for identifying hot edge prediction
class Edge(TypedDict):
    edge_to_bb_id: int
    freq: int

class BasicBlock(TypedDict):
    bb_id: int
    actual_freq: int
    predicted_freq: int
    predicted_edges: List[Edge]
    actual_edges: List[Edge]

class Cfg(TypedDict):
    cfg_bbs: List[BasicBlock]



class Improvement(enum.Enum):
    OK = "Ok"
    ZeroDivisionError = "ZeroDivisionError"
    NameError = "NameError"
    TypeError = "TypeError"
    ValueError = "ValueError"
    IndexError = "IndexError"
    SyntaxError = "SyntaxError"
    PerformanceImprovement = "PerformanceImprovement"
    SafetyImprovement = "SafetyImprovement"

class Improvements(TypedDict):
    type_of_improvement: str
    improvement_desc: str
    start_line: int           
    end_line: int
    
class Response(TypedDict):
    improvement_list: List[Improvements]

class FixSuggestion(TypedDict):
    fix_code: str             
    start_line: int           
    end_line: int
"""Generative AI model integration for GINS
"""
class BaseModel:
    def __init__(self, model_name: str = "gemini-2.0-flash", **kwargs):
        self.model_name = model_name
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.config()

    def config(self):
        raise NotImplementedError

    def generate(self, prompt: str):
        raise NotImplementedError


class Gemini(BaseModel):
    def config(self):
        try:
            import google.generativeai as genai

            if "api_key" in self.__dict__:
                genai.configure(api_key=self.api_key)
            else:
                import os

                genai.configure(api_key="AIzaSyDCb9X9QEPg12gPPcfvAu7zudYnF9Qowu0")
            self.model = genai.GenerativeModel()
            import os
        except Exception as e:
            print(
                "google.generativeai module not present. Please install using 'pip install google.generativeai'."
            )
            print("Warning:", e)
            return None

    def generate(self, prompt: str):
        response = self.model.generate_content(prompt, )
        usage_metadata = response.usage_metadata
        token_info = {
            "prompt_token_count": usage_metadata.prompt_token_count,
            "candidates_token_count": usage_metadata.candidates_token_count,
            "total_token_count": usage_metadata.total_token_count,
        }
        return response.text, token_info
        
    def generate_structured(self, prompt: str):
        import google.generativeai as genai
        response = self.model.generate_content(
          prompt,
          generation_config=genai.GenerationConfig(
          response_mime_type='application/json',
          response_schema=Response),)
        response_dict = json.loads(response.text)
        usage_metadata = response.usage_metadata
        token_info = {
            "prompt_token_count": usage_metadata.prompt_token_count,
            "candidates_token_count": usage_metadata.candidates_token_count,
            "total_token_count": usage_metadata.total_token_count,
        }
        response_dict = json.loads(response.text)
        return response_dict, token_info

    def generate_fixed_code(self, prompt: str):
        import google.generativeai as genai
        response = self.model.generate_content(
          prompt,
          generation_config=genai.GenerationConfig(
          response_mime_type='application/json',
          response_schema=FixSuggestion),)
        response_dict = json.loads(response.text)
        usage_metadata = response.usage_metadata
        token_info = {
            "prompt_token_count": usage_metadata.prompt_token_count,
            "candidates_token_count": usage_metadata.candidates_token_count,
            "total_token_count": usage_metadata.total_token_count,
        }
        response_dict = json.loads(response.text)
        return response_dict, token_info
