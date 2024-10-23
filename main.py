
#           Import 
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fuzzywuzzy import process





'''
Simple LNP Model For Tips Recommendation about illness and symptoms
 
feacture : 
    1-Convert humain text to usefull words 
    2-Handel human error of writing usefull words
    3-Matching the usefull words with dataset
    4-Show tips if there is a good match
    5-libaraies used : 
                        pandas: A powerful data manipulation and analysis library providing data structures and handling structured data.
                        nltk: A popular natural language processing (NLP) package for Python.
                        fuzzywuzzy: A Python library for string matching and comparison.
                        nltk.corpus.stopwords: A collection of common stop words in various languages, used to filter out non-essential words in text processing.
                        nltk.tokenize.word_tokenize: A tokenizer that splits text into individual words and punctuation, facilitating text analysis.


'''

# Global variables 
ThresholdValue = 60  # threshold for a good match


# Load the stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load your symptoms and tips data
symptoms_df = pd.read_csv('symptoms.csv')  # Ensure this file exists with 'Symptom' and 'Symptom_ID' columns
tips_df = pd.read_csv('tips.csv')          # Ensure this file exists with 'Tip' and 'Symptom_ID' columns





# Preprocess the input text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalnum()]  # Keep only alphanumeric tokens
    return tokens

# Get the closest matching symptom
def get_closest_symptom(user_input):
    global ThresholdValue
    symptoms_list = symptoms_df['Symptom'].tolist()
    closest_match = process.extractOne(user_input, symptoms_list)
    if closest_match and closest_match[1] >= ThresholdValue:  # threshold of 80 for a good match
        print("\n",closest_match,"\n")
        return closest_match[0]
    return None

# Get tips based on symptom
def get_tips(symptom):
    symptom_id = symptoms_df.loc[symptoms_df['Symptom'] == symptom, 'Symptom_ID'].values
    if symptom_id.size > 0:
        tips = tips_df[tips_df['Symptom_ID'] == symptom_id[0]]['Tip'].tolist()
        return tips
    return []

# Main function to handle user input
def main():
    user_input = "i feel like i have fiver "
    tokens = preprocess_text(user_input)
    symptom = ' '.join(tokens)

    closest_symptom = get_closest_symptom(symptom)
    
    if closest_symptom:
        print(f"Closest match found: {closest_symptom}")
        tips = get_tips(closest_symptom)
        if tips:
            print("Here are some tips:")
            for tip in tips:
                print(f"- {tip}")
        else:
            print("No tips found for this symptom.")
    else:
        print("No matching symptom found.")

if __name__ == "__main__":
    main()




'''
1-Taskes add more than one tip for each symptom 
2-Make random selection from tips when there is matching symptom
3-find dataset for this 
4-make it matching multi symptoms at same time with ThresholdValue >= 60 
5-bigger dataset of Symptom
6-test results and make it more accurate
7-test the preformans of the code
'''