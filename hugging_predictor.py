from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

class HuggingFaceMoodPredictor:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables. Please set it in your .env file.")
            
        self.client = InferenceClient(
            provider="novita",
            api_key=api_key,
        )
    
    def analyze_mood(self, text):
        """Analyze the mood of the given text."""
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3-0324",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Analyze the emotional state in this text and categorize it as either 'happy', 'sad', or 'neutral'. Consider:
- Word choice and tone
- Context and situation described
- Intensity of emotions expressed
- Underlying sentiment

Text to analyze: '{text}'

Respond with ONLY ONE WORD from the options: 'happy', 'sad', or 'neutral'."""
                    }
                ],
            )
            response = completion.choices[0].message.content.lower().strip()
            confidence = 0.8  # Example confidence value
            return {
                "mood": response,
                "confidence": confidence
            }
        except Exception as e:
            print(f"Error analyzing mood: {str(e)}")
            return None

    def suggest_activities(self, mood, text):
        """Suggest activities based on the detected mood."""
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3-0324",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Based on the user's mood '{mood}' and their words '{text}', suggest 3 unique and creative activities. Consider:
- The intensity and nature of their emotional state
- Activities that could either improve or maintain their current mood
- A mix of indoor and outdoor activities
- Both solo and social options
- Activities that engage different senses and interests
- Unexpected but effective mood-boosting activities

Return ONLY a JSON object in this exact format, without any markdown or additional text:
{{
  "message": "A brief encouraging message tailored to their mood",
  "activities": [
    "First creative and detailed activity suggestion",
    "Second unique and specific activity suggestion",
    "Third unexpected but mood-appropriate activity suggestion"
  ]
}}"""
                    }
                ],
            )
            
            # Extract the response content and clean it
            response_text = completion.choices[0].message.content.strip()
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            try:
                import json
                parsed = json.loads(response_text)
                return parsed
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                return {
                    "message": "Activities suggestion",
                    "activities": [response_text]
                }
                
        except Exception as e:
            print(f"Error suggesting activities: {str(e)}")
            return None
        
    def therapist_response(self, mood, text):
        """Get a short therapeutic response based on the mood and text."""
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3-0324",
                messages=[
                    {
                        "role": "user",
                        "content": f"""As a warm, empathetic therapist, provide a thoughtful response to someone who's feeling {mood}. Their exact words were: '{text}'

Consider in your response:
- Validate their emotions and experiences
- Show genuine understanding and empathy
- Offer gentle perspective or reframing if appropriate
- Include subtle therapeutic techniques (like cognitive reframing, mindfulness, or emotional validation)
- End with a note of hope or encouragement

Provide a brief, 2-3 sentence response that feels personal and supportive. Return only the response without quotes or formatting."""
                    }
                ],
            )
            
            return completion.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"Error getting therapist response: {str(e)}")
            return None
