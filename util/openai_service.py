from openai import OpenAI
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

client = OpenAI(api_key=config['LOCAL']['OPENAI_KEY'])

# Use a model from OpenAI (assuming "text-embedding-ada-002" exists for this example)
model_name="gpt-3.5-turbo-1106"
def analyze_sentiment(prompt):
    """
    Sends the prompt to OpenAI API using the chat interface and gets the model's response.
    """

    system_message = {
        "role": "system",
        "content": "return the sentiments in one word for each review  as : {positive} or {negative} and if contains add {feature improvement} " +
        "example: {positive} {feature improvement} {negative}"
    }
    user_message = {
        'role': 'user',
        'content': prompt
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=[system_message, user_message],
        temperature=0,
        max_tokens=256
    )

    # Extract the chatbot's message from the response.
    # Assuming there's at least one response and taking the last one as the chatbot's reply.
    chatbot_response = response.choices[0].message.content
    print(chatbot_response)
    return chatbot_response

def analyze_label(prompt):
    """
    Sends the prompt to OpenAI API using the chat interface and gets the model's response.
    """

    system_message = {
        "role": "system",
        "content": prompt
    }
   

    response = client.chat.completions.create(
        model=model_name,
        messages=[system_message],
        temperature=0,
        max_tokens=256
    )

    # Extract the chatbot's message from the response.
    # Assuming there's at least one response and taking the last one as the chatbot's reply.
    chatbot_response = response.choices[0].message.content
    return chatbot_response