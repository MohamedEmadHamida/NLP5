#           Import 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process
import random






'''
Simple LNP Model For Tips Recommendation about illness and symptoms
 
feacture : 
    1-Convert humain text to usefull words 
    2-Handel human error of writing usefull words
    3-Matching the usefull words with dataset
    4-Show tips if there is a good match form random tips
    5-multitip return form singel user input 
    libaraies used : 
                        pandas: A powerful data manipulation and analysis library providing data structures and handling structured data.
                        nltk: A popular natural language processing (NLP) package for Python.
                        fuzzywuzzy: A Python library for string matching and comparison.
                        nltk.corpus.stopwords: A collection of common stop words in various languages, used to filter out non-essential words in text processing.
                        nltk.tokenize.word_tokenize: A tokenizer that splits text into individual words and punctuation, facilitating text analysis.


'''
# Global variables 
ThresholdValue = 60  # threshold for a good match help with human errors only 

# Load the stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load your symptoms and tips data
symptoms_df = pd.read_csv('symptoms.csv')  # Ensure this file exists with 'Symptom' and 'Symptom_ID' columns
tips_df = pd.read_csv('tips.csv')          # Ensure this file exists with 'Tip' and 'Symptom_ID' columns


# Preprocess the input text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    #tokens = [word for word in tokens if word.isalnum()]  # Keep only alphanumeric tokens
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return tokens

#list if symptoms matches
def get_closest_symptom(user_input):
    global ThresholdValue
    
    symptoms_list = symptoms_df['Symptom'].tolist()
    closest_matches = process.extract(user_input, symptoms_list)
    
    # Filter matches based on the threshold and retrieve their Symptom_IDs
    matched_symptom_ids = [
        symptoms_df.loc[symptoms_df['Symptom'] == match[0], 'Symptom_ID'].values[0]
        for match in closest_matches if match[1] >= ThresholdValue
    ]
    
    if matched_symptom_ids:
        print("\n", matched_symptom_ids, "\n")
        return matched_symptom_ids
    
    return []

# Get random tips based on symptom
# Get random tips

def get_tips(symptom_ids):
    tips_for_symptoms = {}
    
    for symptom_id in symptom_ids:
        # Get tips from Tip1, Tip2, and Tip3 columns for the given Symptom_ID
        tips = tips_df.loc[tips_df['Symptom_ID'] == symptom_id, ['Tip1', 'Tip2', 'Tip3']].values.flatten().tolist()
        
        # Filter out any None or NaN values
        tips = [tip for tip in tips if pd.notna(tip)]  # Removes None or NaN tips
        
        if tips:
            tips_for_symptoms[symptom_id] = random.choice(tips)  # Choose a random tip for the symptom ID
    
    return tips_for_symptoms

# Function to handle the closest symptom and tips
def handle_closest_symptom(closest_symptom):
    if closest_symptom:
        print(f"Closest match found: {closest_symptom}")
        tip = get_tips(closest_symptom)
        if tip:
            print(f"Tip: {tip}")
        else:
            print("No tips found for this symptom.")
    else:
        print("No matching symptom found.")

# Main function to handle user input
def main():
    user_input = " I have Fiver"
    tokens = preprocess_text(user_input)
    symptom = ' '.join(tokens)

    closest_symptom = get_closest_symptom(symptom)
    handle_closest_symptom(closest_symptom)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
'''
1-Taskes add more than one tip for each symptom  ✓
2-Make random selection from tips when there is matching symptom ✓
3-more than one symptom in the same time ✓
4-find dataset for this 
5-make it matching multi symptoms at same time with ThresholdValue >= 60 ✓
6-bigger dataset of Symptom 
7-test results and make it more accurate
8-test the preformans of the code
9-super loop with one time load data 
'''

'''
NLP (Natural Language Processing) models can perform a range of tasks, including:

Text Classification - Categorize text by sentiment, topic, or intent.
Named Entity Recognition (NER) – Identify entities like names, locations, or dates in text.
Text Summarization – Condense long texts into shorter summaries.
Language Translation – Translate text between languages.
Speech Recognition – Convert spoken words into text.
Chatbots and Conversational AI – Enable automated, natural-sounding interactions.
Sentiment Analysis – Detect emotions and opinions in text.
Text Generation – Generate coherent text based on input prompts.
Spell and Grammar Correction – Correct grammatical and spelling errors in text.
Intent Recognition – Identify user intent to trigger actions (useful for health or customer support apps, for instance).


'''