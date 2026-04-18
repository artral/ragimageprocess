import os
import base64
import requests


class ImageProcessor:
    """RAG-based image processor that tags images using approved naming conventions."""
    
    # 1. THE RETRIEVAL STEP (Simulated Knowledge Base)
    # In a large-scale RAG, you'd use a vector DB like ChromaDB or Pinecone.
    VALID_TAGS = ["industrial_pump_v1", "circuit_board_rev_A", "safety_valve_44"]
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, api_key):
        """Initialize the ImageProcessor with an API key."""
        self.api_key = api_key
    
    def get_specific_naming_context(self):
        """Retrieves the list of approved tags to guide the AI."""
        return ", ".join(self.VALID_TAGS)
    
    # 2. PREPARING THE IMAGE
    def encode_image(self, image_path):
        """Converts a JPG to a Base64 string for the API."""
        with open(image_path, "rb") as img:
            return base64.b64encode(img.read()).decode('utf-8')
    
    # 3. THE GENERATION STEP (Multimodal AI)
    def tag_image(self, image_path):
        """Tags a single image using the approved naming context."""
        base64_image = self.encode_image(image_path)
        tag_context = self.get_specific_naming_context()  # The 'Retrieval' part

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # The 'Augmented' prompt ensures the AI stays grounded to your data.
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Select the best tag from this list: {tag_context}. Return ONLY the tag."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        }

        response = requests.post(self.OPENAI_API_URL, headers=headers, json=payload)
        print(f"API Response: {response.status_code} - {response.text}")  # Debug print
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        return response.json()['choices'][0]['message']['content']
    
    # 4. PROCESSING THE FOLDER
    def process_folder(self, folder_path):
        """Processes all JPG images in a folder and tags them."""
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist.")
            return
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                full_path = os.path.join(folder_path, filename)
                try:
                    tag = self.tag_image(full_path)
                    print(f"Tagged {filename} as: {tag}")
                except Exception as e:
                    print(f"Error tagging {filename}: {e}")
