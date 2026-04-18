import unittest
import json
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_tagger_gemini import ImageProcessor_Gemini

class TestParseJsonResponse(unittest.TestCase):
    """Unit tests for ImageProcessor_Gemini.parse_json_response method"""

    def setUp(self):
        """Initialize test fixtures"""
        self.processor = ImageProcessor_Gemini(api_key="test_key")
        self.valid_json = {
            "category": "food",
            "tags": [
                "pizza",
                "pepperoni",
                "sausage",
                "mushrooms",
                "olives"
            ],
            "description": "A whole pizza topped with a variety of ingredients including pepperoni, sausage, mushrooms, green peppers, and black olives.",
            "confidence": "0.99"
}

    def test_valid_json_parsing(self):
        """Test parsing valid JSON string"""
        json_str = json.dumps(self.valid_json)
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result, self.valid_json)

    def test_json_with_markdown_fences(self):
        """Test parsing JSON wrapped in markdown code fences"""
        json_str = f"```json\n{json.dumps(self.valid_json)}\n```"
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result, self.valid_json)

    def test_json_with_generic_markdown_fences(self):
        """Test parsing JSON with generic markdown fences"""
        json_str = f"```\n{json.dumps(self.valid_json)}\n```"
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result, self.valid_json)

    def test_json_with_escaped_newlines(self):
        """Test parsing JSON with escaped newline characters"""
        json_str = json.dumps(self.valid_json).replace("\n", "\\n")
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result, self.valid_json)

    def test_json_with_extra_whitespace(self):
        """Test parsing JSON with leading/trailing whitespace"""
        json_str = f"   \n{json.dumps(self.valid_json)}\n   "
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result, self.valid_json)

    def test_json_with_missing_closing_brace(self):
        """Test parsing JSON with missing closing brace"""
        json_str = json.dumps(self.valid_json)[:-1]  # Remove last }
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result, self.valid_json)

    def test_json_with_multiple_missing_closing_braces(self):
        """Test parsing JSON with multiple missing closing braces"""
        json_str = json.dumps(self.valid_json)[:-2]  # Remove last 2 characters
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result, self.valid_json)

    def test_json_with_markdown_and_missing_brace(self):
        """Test parsing JSON with markdown fences and missing closing brace"""
        json_str = f"```json\n{json.dumps(self.valid_json)[:-1]}\n```"
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result, self.valid_json)

    def test_partial_json_regex_fallback(self):
        """Test regex fallback when JSON is partially malformed"""
        # Valid JSON embedded in text
        json_str = f"Some text before {json.dumps(self.valid_json)} some text after"
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result, self.valid_json)

    def test_manual_field_extraction_fallback(self):
        """Test manual field extraction when all else fails"""
        # Malformed JSON that can still be extracted manually
        malformed = '''
        {
            "category": "animal",
            "tags": ["dog", "cute", "fluffy"],
            "description": "A cute golden retriever",
            "confidence": 0.88
        '''
        result = self.processor.parse_json_response(malformed)
        self.assertEqual(result["category"], "animal")
        self.assertEqual(result["tags"], ["dog", "cute", "fluffy"])
        self.assertEqual(result["description"], "A cute golden retriever")
        self.assertEqual(result["confidence"], 0.88)

    def test_partial_fields_extraction(self):
        """Test manual extraction with only some fields present"""
        partial = '''
        Some response text here
        "category": "vehicle"
        "tags": ["car", "red", "fast"]
        More text
        '''
        result = self.processor.parse_json_response(partial)
        self.assertEqual(result["category"], "vehicle")
        self.assertEqual(result["tags"], ["car", "red", "fast"])

    def test_confidence_float_conversion(self):
        """Test that confidence is converted to float"""
        json_str = json.dumps(self.valid_json)
        result = self.processor.parse_json_response(json_str)
        self.assertIsInstance(result["confidence"], float)
        self.assertEqual(result["confidence"], 0.95)

    def test_confidence_as_string_number(self):
        """Test manual extraction of confidence as string number"""
        response = 'confidence": 0.75'
        result = self.processor.parse_json_response(response)
        self.assertEqual(result["confidence"], 0.75)

    def test_tags_as_list(self):
        """Test that tags are returned as a list"""
        json_str = json.dumps(self.valid_json)
        result = self.processor.parse_json_response(json_str)
        self.assertIsInstance(result["tags"], list)
        self.assertEqual(len(result["tags"]), 3)

    def test_empty_tags_array(self):
        """Test handling of empty tags array"""
        data = self.valid_json.copy()
        data["tags"] = []
        json_str = json.dumps(data)
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result["tags"], [])

    def test_special_characters_in_description(self):
        """Test handling description with special characters"""
        data = self.valid_json.copy()
        data["description"] = 'A pizza with "quotes" and special chars: !@#$%'
        json_str = json.dumps(data)
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result["description"], data["description"])

    def test_unicode_in_fields(self):
        """Test handling Unicode characters"""
        data = self.valid_json.copy()
        data["description"] = "Un délicieux pizza 🍕"
        data["tags"] = ["pizza", "français", "😋"]
        json_str = json.dumps(data, ensure_ascii=False)
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result["description"], data["description"])

    def test_completely_invalid_input_raises_error(self):
        """Test that completely invalid input raises ValueError"""
        invalid_text = "This is just plain text with no JSON structure at all"
        with self.assertRaises(ValueError):
            self.processor.parse_json_response(invalid_text)

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError"""
        with self.assertRaises(ValueError):
            self.processor.parse_json_response("")

    def test_only_whitespace_raises_error(self):
        """Test that only whitespace raises ValueError"""
        with self.assertRaises(ValueError):
            self.processor.parse_json_response("   \n\t   ")

    def test_json_with_extra_fields(self):
        """Test parsing JSON with additional fields"""
        data = self.valid_json.copy()
        data["extra_field"] = "should be ignored"
        data["nested"] = {"key": "value"}
        json_str = json.dumps(data)
        result = self.processor.parse_json_response(json_str)
        # Should still have the main fields
        self.assertEqual(result["category"], self.valid_json["category"])

    def test_category_case_sensitivity(self):
        """Test that category is extracted as-is (case sensitive)"""
        data = self.valid_json.copy()
        data["category"] = "FOOD"
        json_str = json.dumps(data)
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result["category"], "FOOD")

    def test_tags_with_quotes_in_values(self):
        """Test extracting tags that contain escaped quotes"""
        json_str = '{"tags": ["tag\\"with\\"quotes", "normal_tag"]}'
        result = self.processor.parse_json_response(json_str)
        # Should extract the tags (manual fallback)
        self.assertIn("tags", result)

    def test_multiline_description(self):
        """Test handling multiline descriptions"""
        data = self.valid_json.copy()
        data["description"] = "Line 1\\nLine 2\\nLine 3"
        json_str = json.dumps(data)
        result = self.processor.parse_json_response(json_str)
        self.assertIn("Line 1", result["description"])

    def test_confidence_zero(self):
        """Test parsing confidence value of 0"""
        data = self.valid_json.copy()
        data["confidence"] = 0.0
        json_str = json.dumps(data)
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result["confidence"], 0.0)

    def test_confidence_one(self):
        """Test parsing confidence value of 1"""
        data = self.valid_json.copy()
        data["confidence"] = 1.0
        json_str = json.dumps(data)
        result = self.processor.parse_json_response(json_str)
        self.assertEqual(result["confidence"], 1.0)

    def test_partial_response_missing_category(self):
        """Test manual extraction with missing category"""
        response = '''
        "tags": ["item1", "item2"],
        "description": "A description",
        "confidence": 0.5
        '''
        result = self.processor.parse_json_response(response)
        self.assertNotIn("category", result)
        self.assertEqual(result["tags"], ["item1", "item2"])

    def test_partial_response_missing_tags(self):
        """Test manual extraction with missing tags"""
        response = '''
        "category": "person",
        "description": "A person",
        "confidence": 0.9
        '''
        result = self.processor.parse_json_response(response)
        self.assertEqual(result["category"], "person")
        self.assertNotIn("tags", result)


class TestImageProcessorGeminiIntegration(unittest.TestCase):
    """Integration tests for ImageProcessor_Gemini"""

    def setUp(self):
        """Initialize test fixtures"""
        self.processor = ImageProcessor_Gemini(api_key="test_key")

    def test_processor_initialization(self):
        """Test that processor initializes correctly"""
        self.assertIsNotNone(self.processor.api_key)
        self.assertIsNotNone(self.processor.client)
        self.assertIsNotNone(self.processor.PROMPT)

    def test_processor_has_parse_json_response_method(self):
        """Test that processor has parse_json_response method"""
        self.assertTrue(hasattr(self.processor, "parse_json_response"))
        self.assertTrue(callable(getattr(self.processor, "parse_json_response")))


if __name__ == "__main__":
    unittest.main()
