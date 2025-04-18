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



class Error(enum.Enum):
    OK = "Ok"
    ZeroDivisionError = "ZeroDivisionError"
    NameError = "NameError"
    TypeError = "TypeError"
    ValueError = "ValueError"
    IndexError = "IndexError"
    SyntaxError = "SyntaxError"

class ErrorResponse(TypedDict):
    type_of_error: Error
    error_line_number: str


class Response(TypedDict):
    behavior_description : str
    error_list: List[ErrorResponse]
"""Generative AI model integration for GINS
"""
class BaseModel:
    def __init__(self, model_name: str = "gemini-1.5-pro", **kwargs):
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

                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
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
        return response.text
        
    def generate_structured(self, prompt: str):
        import google.generativeai as genai
        response = self.model.generate_content(
          prompt,
          generation_config=genai.GenerationConfig(
          response_mime_type='application/json',
          response_schema=Response),)
        response_dict = json.loads(response.text)
        return response_dict
    
    def fix_errors(self, suggestions, original_text):
        import google.generativeai as genai
        prompt = f"""
            Please fix the following errors in this file: {suggestions}. The original file is: {original_text}.
            Respond with raw code only — no markdown, no triple quotes.
            """
        response = self.model.generate_content(prompt, )
        return response.text
