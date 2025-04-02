# MoodTracker

A machine learning-based mood tracking system that uses HuggingFace's AI models to analyze moods, provide therapeutic insights, and suggest activities.

## Features

- AI-powered mood analysis
- Therapeutic response generation
- Personalized activity suggestions
- Mood pattern analysis and visualization
- Interactive GUI with Tkinter

## Requirements

All required packages are listed in `requirements.txt`. Main dependencies include:
- huggingface-hub
- pandas
- matplotlib
- seaborn
- tkinter (usually comes with Python)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Set up your environment:
   - Copy `.env.template` to a new file named `.env`
   - Sign up for a HuggingFace account at https://huggingface.co
   - Get your API key from https://huggingface.co/settings/tokens
   - Replace `your_api_key_here` in `.env` with your actual API key

## Usage

Run the application:
```bash
python app.py
```

## Security Note

Never commit your `.env` file or share your API key. The `.env` file is already added to `.gitignore` to prevent accidental commits.

## License

MIT License
