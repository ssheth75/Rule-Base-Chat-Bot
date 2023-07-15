import nltk
import random
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.chat.util import Chat, reflections
from nltk.corpus import wordnet

# Chatbot's rules
chatbot_rules = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'what is your name?', ['I am a chatbot.', 'You can call me Chatbot.']),
    (r'how are you?', ['I am doing well, thank you!', 'I\'m fine, thanks for asking.']),
    (r'my name is (.*)', ['Hello %1! Pleased to meet you.', 'Hi %1! How can I assist you?']),
    (r'what can you do\?', ['I can answer questions, provide information, or just have a friendly conversation.']),
    (r'weather', ['Sorry, I cannot provide real-time weather information.']),
    (r'(.*) (happy|good|great)', ['That\'s wonderful!', 'I\'m glad to hear that.', 'Great!']),
    (r'(.*) (sad|bad|not good)', ['I\'m sorry to hear that.', 'That\'s unfortunate.', 'I hope things get better.']),
    (r'quit', ['Goodbye!', 'It was nice talking to you. Goodbye!']),
]

# Create a chatbot instance
chatbot = Chat(chatbot_rules, reflections)

# Preprocess user input
def preprocess_input(input_text):
    tokens = nltk.word_tokenize(input_text)
    tokens = [token.lower() for token in tokens]
    return tokens

# Generate a random greeting
def generate_random_greeting():
    greetings = ['Hello!', 'Hi there!', 'Greetings!', 'Hey!']
    return random.choice(greetings)

# Get synonyms for a word
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)

# Perform sentiment analysis
def analyze_sentiment(input_text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(input_text)
    compound_score = sentiment_scores['compound']
    return compound_score

# Start the conversation
print("Chatbot:", generate_random_greeting(), "How can I assist you today?")
previous_input = ""
while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        break
    else:
        preprocessed_input = preprocess_input(user_input)
        sentiment_score = analyze_sentiment(user_input)

        if sentiment_score >= 0.5:
            response = "I'm glad you're feeling positive!"
        elif sentiment_score <= -0.5:
            response = "I'm sorry to hear that you're feeling negative."

        if not response:
            response = chatbot.respond(preprocessed_input)

        if not response:
            # Generate a response using synonyms
            synonyms = set()
            for token in preprocessed_input:
                token_synonyms = get_synonyms(token)
                synonyms = synonyms.union(token_synonyms)
            if synonyms:
                response = "Are you referring to " + ", ".join(synonyms) + "?"

        if not response:
            response = "I'm sorry, I don't understand. Can you please rephrase your question?"

        if response == previous_input:
            response = "You already said that. Can we talk about something else?"

        previous_input = user_input

        print("Chatbot:", response)
