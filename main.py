#           Import 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process
import random


DEBUG = False  # Set to False to disable all print statements


def debug_print(message):
    if DEBUG:
        print('\n',message,'\n')



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
    debug_print(tokens)
    return tokens








# Get closest symptom, including synonyms
def get_closest_symptom(user_input):
    global ThresholdValue
    
    # Create a list of symptoms and their synonyms
    symptoms_list = symptoms_df['Symptom'].tolist()
    synonyms_list = symptoms_df['Synonyms'].dropna().tolist()
    
    # Combine symptoms with their synonyms
    all_symptoms = symptoms_list + [syn for synonyms in synonyms_list for syn in synonyms.split(';')]
    
    closest_matches = process.extract(user_input, all_symptoms)
    
    # Filter matches based on the threshold and retrieve their original DataFrame indices
    matched_indices = [
        symptoms_df.index[symptoms_df['Symptom'] == match[0]][0]  # Returns the first match if multiple exist
        for match in closest_matches if match[1] >= ThresholdValue
    ]
    
    if matched_indices:
        print("\n", matched_indices, "\n")
        return matched_indices
    
    return []






# Get random tips based on matched indices
def get_tips(matched_indices):
    tips_for_symptoms = {}
    
    # Get all corresponding Symptom_IDs from the matched indices
    symptom_ids = symptoms_df.loc[matched_indices, 'Symptom_ID'].tolist()
    
    # Get tips for the list of Symptom_IDs
    for symptom_id in set(symptom_ids):  # Use set to avoid duplicates
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
        debug_print(f"Closest match found: {closest_symptom}")
        tip = get_tips(closest_symptom)
        if tip:
            print("\n\n\n",f"Tip: {tip}","\n\n\n")
        else:
            print("No tips found for this symptom.")
    else:
        print("No matching symptom found.")

# Main function to handle user input
def main():
    user_input = " I have Fever "
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