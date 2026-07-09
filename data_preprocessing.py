import re
from nltk.stem import WordNetLemmatizer

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
    text = re.sub(r'^.*?\s*\(Reuters\)\s*-\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\s*\S+|www\.\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) #remove punctuation
    return re.sub(r'\s+', ' ', text).strip()

def lemmatize(text):
    words = str(text).split()
    lemmatized = []
    for w in words:
        lemma = lemmatizer.lemmatize(w.lower())
        
        #keep the uppercase signal
        if w.isupper() and len(w) > 1:
            lemmatized.append(lemma.upper())
        else:
            lemmatized.append(lemma)
            
    return " ".join(lemmatized)


