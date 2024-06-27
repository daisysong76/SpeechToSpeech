import string
import unicodedata
import inflect

# Initialize inflect engine for number to word conversion
p = inflect.engine()

# Dictionary for contractions and abbreviations

contractions_dict = {
    "don't": "do not",
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'ll": " will",
    "'ve": " have",
    "'re": " are",
    "'d": " would",
    "it's": "it is",
    "i'm": "i am",
    "let's": "let us",
    "she's": "she is",
    "he's": "he is",
    "they're": "they are",
    "we're": "we are",
    "there's": "there is",
    "here's": "here is",
    "who's": "who is",
    "what's": "what is",
    "that's": "that is",
    "how's": "how is",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "they'd": "they would",
    "we'd": "we would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "they'll": "they will",
    "we'll": "we will",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
    "we've": "we have",
    "y'all": "you all",
    "n't": " not",
    "'s": " is",
    "'m": " am",
    "'re": " are",
    "'d": " would",
    "'ve": " have",
    "'ll": " will"
    # Add more as needed
}


def expand_contractions(text):
    for contraction, full_form in contractions_dict.items():
        text = text.replace(contraction, full_form)
    return text

def normalize_numbers(text):
    words = text.split()
    normalized_words = []
    for word in words:
        if word.isdigit():
            word = p.number_to_words(word)
        normalized_words.append(word)
    return ' '.join(normalized_words)

def normalize_text(text):
    """
    Normalize the text by converting to lowercase, removing punctuation,
    handling whitespace, normalizing unicode characters, expanding contractions,
    and converting numbers to words.
    """
    # Convert to lowercase
    text = text.lower()
    # Normalize unicode characters (e.g., accented characters)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip leading and trailing whitespaces
    text = text.strip()
    # Replace all types of whitespace characters with a single space
    text = ' '.join(text.split())
    # Expand contractions and abbreviations
    text = expand_contractions(text)
    # Normalize numbers
    text = normalize_numbers(text)
    return text
