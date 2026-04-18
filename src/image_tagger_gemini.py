import base64
import json
import os
from unittest import result
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

from pathlib import Path
import re

from io import BytesIO

load_dotenv()  # Load environment variables from .env file


class ImageProcessor_Gemini:
    def __init__(self, api_key):
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "GEMINI_API_KEY"))
        self.PROMPT = """Analyze this image and respond ONLY with a valid JSON object.
                    No markdown, no code fences, no explanation — raw JSON only.

                    {
                    "category": "one of: food, animal, vehicle, person, nature, object",
                    "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
                    "description": "one sentence description",
                    "confidence": 0.95
                    }"""

        
    def process_image(self, image_path):
        from PIL import Image

        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        else:
            return Image.open(image_path)

    def classify_image(self, image_path):
        """Classify an image and return structured tags."""
        image = self.process_image(image_path)

        response = self.client.models.generate_content(
                         model="gemini-3-flash-preview",        # ✅ current free-tier model
                         contents=[self.PROMPT, image],
                         config=types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=300,
                            )
                        )

        raw = response.text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        result = self.parse_json_response(raw)
        result["source"] = image_path
        return result
    
    def parse_json_response(self,text: str) -> dict:
        # 1. Strip markdown fences
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        
        # 2. Replace literal \n characters (not real newlines)
        text = text.replace("\\n", "\n")
        
        # 3. Strip whitespace
        text = text.strip()
        
        # 4. Fix missing closing brace (model sometimes cuts off)
        open_braces = text.count("{")
        close_braces = text.count("}")
        if open_braces > close_braces:
            text += "}" * (open_braces - close_braces)
        
        # 5. Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 6. Extract JSON object as fallback
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        
        # 7. Last resort — manually extract fields
        result = {}
        
        category = re.search(r'"category"\s*:\s*"([^"]+)"', text)
        if category:
            result["category"] = category.group(1)
        
        tags = re.search(r'"tags"\s*:\s*\[([^\]]+)\]', text)
        if tags:
            result["tags"] = re.findall(r'"([^"]+)"', tags.group(1))
        
        desc = re.search(r'"description"\s*:\s*"([^"]+)"', text)
        if desc:
            result["description"] = desc.group(1)
        
        conf = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
        if conf:
            result["confidence"] = float(conf.group(1))
        
        if result:
            return result
        
        raise ValueError(f"Could not parse response: {text}")

    def is_food(self, source: str) -> bool:
        """Quick check: is this image food?"""
        result = self.classify_image(source)
        is_food = result.get("category", "").lower() == "food"
        print(f"Image: {source} | Category: {result.get('category')} | Is Food? {is_food}")
        return is_food
    