import random

# Predefined responses
responses = {
    "hello": ["Hi there!", "Hello!", "Hey! 👋"],
    "hi": ["Hi!", "Any help!!!", "Hey! 👋"],
    "how are you": ["I'm fine 😊", "Doing great!", "All good!"],
    "bye": ["Goodbye!", "See you later!", "Bye! 👋"],
    "name": ["I'm your chatbot 🤖", "You can call me NLP Bot"],
    "llm":["Large Language Model", "AI system trained on vast datasets","Understand, generate, and summarize human-like text"],
    "generative ai":["Generative AI (GenAI) is a type of artificial intelligence that creates new content—including text, images, video, audio, and software code—by learning patterns from existing data"],
"good":["Good Morning and how are you today","Good morning. Any help?"],
"morning":["GM","HRU help?"]

}

def get_response(user_input):
    user_input = user_input.lower()

    for key in responses:
        if key in user_input:
            return random.choice(responses[key])

    return "Sorry, I don't understand 🤔"