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

from scipy import io

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
        import io

        if image_path.startswith("http://") or image_path.startswith("https://"):
            response = requests.get(image_path)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        else:
            return Image.open(image_path)

    def classify_image(self, image_path):
        """Classify an image and return structured tags."""
        image = self.process_image(image_path)

        response = self.client.models.generate_content(
                         model="gemini-2.0-flash",        # ✅ current free-tier model
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
    
    def parse_json_response(text: str) -> dict:
        """Safely extract JSON from model response."""
        # Strip markdown fences if model ignored instructions
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # Find JSON object in response
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No valid JSON found in response: {text}")


    def is_food(self, source: str) -> bool:
        """Quick check: is this image food?"""
        result = self.classify_image(source)
        is_food = result.get("category", "").lower() == "food"
        print(f"Image: {source} | Category: {result.get('category')} | Is Food? {is_food}")
        return is_food
    