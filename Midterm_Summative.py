import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import os
import random

# Load dataset
df = pd.read_csv('./conversational_data_with_entities.csv')

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=df['Intent'].nunique())

# Encode intents as numeric labels
label_encoder = LabelEncoder()
df['Intent'] = label_encoder.fit_transform(df['Intent'])

# Tokenize inputs
tokens = tokenizer.batch_encode_plus(df['User Input'].tolist(), padding=True, truncation=True, return_tensors="pt")

# Prepare data
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    tokens['input_ids'], torch.tensor(df['Intent'].values), test_size=0.2)

train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Fine-tune the model
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in train_loader:
        b_input_ids, b_labels = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids, labels=b_labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Load the NER model from transformers
ner_pipeline = pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english')

# Initialize a dictionary to hold possible responses for each intent
response_dict = {
    "greeting": ["Hello! How can I assist you?", "Hi there! What can I do for you?", "Greetings! How may I help you?"],
    "weather_query": ["The weather is sunny today!", "It's raining now.", "Expect cloudy weather later."],
    "flight_booking": ["I can help you book a flight to your destination.", "Let's get started with your flight booking.", "I can assist with flight reservations."],
    "tell_joke": ["Why don't scientists trust atoms? Because they make up everything!", "What did the ocean say to the beach? Nothing, it just waved!", "Why did the scarecrow win an award? Because he was outstanding in his field!"],
    "set_reminder": ["Reminder set!", "Your reminder has been noted.", "I'll remind you as requested."]
}

# Function to save feedback to a CSV file
def save_feedback(user_input, intent, response, rating):
    feedback_file = 'feedback_data.csv'
    
    # Check if the file exists, if not, create it with headers
    if not os.path.exists(feedback_file):
        with open(feedback_file, 'w') as f:
            f.write('User Input,Intent,Response,Rating\n')
    
    # Append the new feedback to the file
    with open(feedback_file, 'a') as f:
        f.write(f"{user_input},{intent},{response},{rating}\n")

# Function to classify intent and extract entities
def classify_intent_and_extract_entities(user_input):
    # Tokenize user input
    inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True)
    
    # Predict intent
    model.eval()
    with torch.no_grad():
        logits = model(inputs['input_ids']).logits
        intent_id = torch.argmax(F.softmax(logits, dim=1), dim=1).item()
        intent = label_encoder.inverse_transform([intent_id])[0]
    
    # Extract entities
    entities = ner_pipeline(user_input)
    
    return intent, entities

# Example usage:
while True:
    # Get user input
    user_input = input("Enter your query: ")
    
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    # Classify intent and extract entities
    intent, entities = classify_intent_and_extract_entities(user_input)
    
    # Select a response based on intent
    possible_responses = response_dict.get(intent, ["I'm sorry, I didn't understand that."])
    
    if possible_responses:
        response = random.choice(possible_responses)
    else:
        response = "I'm sorry, I didn't understand that."
    
    # Display the response
    print(f"Response: {response}")
    
    # Ask for user feedback
    rating = input("Please rate the response (1-5): ")
    
    # Save the feedback
    save_feedback(user_input, intent, response, rating)

    # Reinforcement Learning Adjustment
    if rating in ['1', '2']:  # Low rating
        # Remove the response from possible responses
        if response in response_dict[intent]:
            response_dict[intent].remove(response)
            print(f"The response has been removed due to low rating.")
    elif rating in ['4', '5']:  # High rating
        # Retain the response (no action needed, already retained)
        print(f"Thank you for the positive feedback!")

    # Display the intent and entities
    print(f"Intent: {intent}")
    print(f"Entities: {entities}")

# Optionally save the feedback for further analysis
# This can be expanded with more complex learning algorithms in future iterations.
