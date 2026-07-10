import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()
#Data preprocessing

# 1. merge title and text ---> text
def prepare_X(df):
    X_unprocessed = df.drop(columns=['label', 'subject', 'date'], errors='ignore')
    X_unprocessed['text'] = X_unprocessed['title'] + " " + X_unprocessed['text']
    # drops dupilcates within the dataframe
    X_unprocessed = X_unprocessed.drop_duplicates(subset='text')
    X = X_unprocessed.drop(columns=['title'])
    return X

# 2. prepare labels
def get_labels(df, target_index):
    if 'label' in df.columns:
        return df.loc[target_index, 'label'].reset_index(drop=True)
    return None


# Data cleaning after the train_test_split

def clean_text(text):
    text = re.sub(r'^.*?\s*\(Reuters\)\s*-\s*', ' ', text, flags=re.IGNORECASE) #remove "Reuters" at the beginning
    text = re.sub(r'https?://\s*\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) #remove punctuation
    return re.sub(r'\s+', ' ', text).strip().lower() #after keeping uppercas didn't do much performance-wise: lowercase


def lemmatize(text):
    words = [w for w in str(text).split()]
    # pos-tagging
    words_with_tags = pos_tag(words)
    
    lemmatized_tokens = [
        lemmatizer.lemmatize(w, pos='a' if p[0] == "J" else 'v' if p[0] == "V" else 'r' if p[0] == "R" else 'n') 
        for w, p in words_with_tags
    ]
    return " ".join(lemmatized_tokens)


