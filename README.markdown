# AI-Powered Mental Health Support Chatbot


## Overview

This repository contains the implementation of a command-line interface (CLI)-based AI-powered mental health support chatbot developed as part of a capstone project. The chatbot leverages natural language processing (NLP) to:
- **Detect Sentiment**: Classify user inputs as positive, neutral, or negative using a fine-tuned DistilBERT model.
- **Identify Crises**: Detect potential crisis situations (e.g., suicidal ideation) using a fine-tuned DistilBERT model and provide helpline resources.
- **Generate Responses**: Produce empathetic and contextually relevant responses using a fine-tuned DialoGPT model.

The project aims to provide accessible mental health support through a lightweight CLI interface, suitable for testing and deployment in resource-constrained environments. It was developed using the `nbertagnolli/counsel-chat` dataset from Hugging Face, ensuring ethical handling of sensitive user inputs.

**Disclaimer**: This chatbot is a prototype and not a substitute for professional mental health care. Always seek help from qualified professionals in crisis situations.

## Problem Statement

Access to timely and empathetic mental health support is limited, particularly in high-stress environments. Many individuals struggle to express emotions or seek help during crises, and manual intervention by professionals is not always feasible. This project addresses the challenge of developing an AI-powered chatbot that accurately detects user sentiment, identifies crisis situations, and provides supportive responses via a CLI, enabling rapid deployment and testing.

## Features

- **Sentiment Analysis**: Classifies user inputs into positive, neutral, or negative sentiments (F1 score: 0.85).
- **Crisis Detection**: Identifies crisis indicators with high accuracy (F1 score: 0.90) and suggests professional resources.
- **Response Generation**: Generates empathetic responses tailored to user sentiment, achieving 80% relevance in tests.
- **CLI Interface**: Simple command-line interface for user interaction, accessible in Colab or local environments.
- **Robust Error Handling**: Handles invalid inputs and model errors gracefully.

## Prerequisites

- **Hardware**: GPU (NVIDIA CUDA-enabled) recommended for faster inference; CPU supported.
- **Software**:
  - Python 3.8 or higher
  - PyTorch 2.6.0+cu124
  - Transformers 4.53.1 (Hugging Face)
  - Pandas 2.2.2
  - SpaCy 3.8.7
- **Dataset**: `nbertagnolli/counsel-chat` dataset from Hugging Face.
- **Pre-trained Models**: Fine-tuned models (`sentiment_model_improved`, `crisis_model`, `dialogpt_finetuned`) must be available or trained.

## Installation

### Option 1: Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook and copy the contents of the file given into a cell.
3. Install dependencies:
   ```bash
   !pip install torch==2.6.0+cu124 transformers==4.53.1 pandas==2.2.2 spacy==3.8.7
   !python -m spacy download en_core_web_sm
   ```
4. Ensure pre-trained models are available in `/content/`, or can be trained manually while running the code:
   - `sentiment_model_improved`
   - `crisis_model`
   - `dialogpt_finetuned`
   If missing, train the models and check for your directory path name.
5. Run the cell to start the chatbot.

### Option 2: Local Environment
1. Clone the repository:
   ```bash
   git clone https://github.com/EasyIce667/AI-Powered-Mental-Health-Support-Chatbot-for-Crisis-Intervention.git
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install torch==2.6.0+cu124 transformers==4.53.1 pandas==2.2.2 spacy==3.8.7
   python -m spacy download en_core_web_sm
   ```
4. Place pre-trained models in the `models/` directory or train them using mentioned in the file.

## Usage

1. **Run the Chatbot**:
   The script will:
   - Run test cases to demonstrate functionality.
   - Start the CLI, prompting for user input.

2. **Interact with the Chatbot**:
   - Enter a message at the `Your Message:` prompt.
   - Type `exit` to quit.
   - Example:
     ```
     Mental Health Support Chatbot
     Share your thoughts or feelings, and I'll try to help. Type 'exit' to quit.
     If you're in crisis, I'll suggest resources.

     Your Message: I'm feeling really down today
     Response: I'm sorry to hear that. I’m here for you. Would you like some coping strategies, like deep breathing or journaling?
     ```

3. **Test Cases**:
   The script includes test cases to verify functionality:
   ```python
   test_texts = [
       "I’m really struggling with anxiety",
       "I just got promoted and I’m thrilled",
       "What are some ways to cope with sadness?",
       "Life feels meaningless sometimes",
       "I’m okay, just navigating some challenges",
       "I’m thinking about ending my life"
   ]
   ```
   Expected outputs include sentiment tailored responses and crisis helpline suggestions.

## Training

To train the models (if not using pre-trained ones):
1. Run `train_models.py`:
   ```bash
   python train_models.py
   ```
2. The script:
   - Loads the `nbertagnolli/counsel-chat` dataset.
   - Preprocesses data using SpaCy.
   - Fine-tunes:
     - `distilbert-base-uncased` for sentiment (600 samples, 5 epochs).
     - `distilbert-base-uncased` for crisis detection (500 samples, 5 epochs).
     - `microsoft/DialoGPT-medium` for response generation (2000 samples, 4 epochs).
   - Saves models to `models/`.
3. Ensure GPU availability for faster training.

**Note**: Training requires significant computational resources. Use Google Colab with a GPU runtime if needed.

## Results

- **Sentiment Analysis**: Achieved an F1 score of 0.85 on a test set, accurately classifying sentiments (e.g., "I’m really struggling with anxiety" as negative).
- **Crisis Detection**: Achieved an F1 score of 0.90, correctly identifying crisis inputs (e.g., "I’m thinking about ending my life").
- **Response Generation**: 80% of generated responses were relevant and empathetic, per manual evaluation.
- **CLI Performance**: The CLI interface is lightweight, with response times under 2 seconds on a CPU.

## Ethical Considerations

- The chatbot prioritizes crisis detection to ensure users in distress receive helpline information.
- It is not a replacement for professional mental health care and should be used responsibly.
- User inputs are processed locally and not stored, ensuring privacy.

## Future Improvements that can be made

- Incorporate larger datasets for improved model robustness.
- Add multilingual support for broader accessibility.
- Implement advanced models (e.g., LLaMA) for better response generation.
- Explore graphical interfaces (e.g., web or mobile apps).
- Integrate real-time feedback for continuous model improvement.

## References

- Hugging Face. (2025). Transformers Library Documentation. [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- nbertagnolli. (2024). Counsel-Chat Dataset. [https://huggingface.co/datasets/nbertagnolli/counsel-chat](https://huggingface.co/datasets/nbertagnolli/counsel-chat)
- Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04805*.
- Liu, Y., et al. (2020). A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*.



