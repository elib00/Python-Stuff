import random
import json
import torch
from training_model import NeuralNetwork
from nltk_utils import bag_of_words, tokenize
import speech_recognition as sr
import pyttsx3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("intents.json", "r") as f:
        intents = json.load(f)
        
    FILE = "data.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    word_collection = data["word_collection"]
    tags = data["tags"]
    model_state = data["model_state"]
        
    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    bot_name = "ChatBot"
    print("Let's chat! Say 'quit' or 'end' to exit...")
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    
    chatting = True
    recognizer = sr.Recognizer()
    while chatting:
        value_error_message = "Unable to recognize speech. Please try again."
        
        try:
            with sr.Microphone() as source:
                print("Speak something...")
                recognizer.adjust_for_ambient_noise(source, duration = 1)
                audio = recognizer.listen(source)
                
                input_sentence = recognizer.recognize_google(audio)
                print(f"You: {input_sentence}")
                input_sentence = input_sentence.lower()
                
        except sr.UnknownValueError:
            print(value_error_message)
            engine.say(value_error_message)
            engine.runAndWait()
            continue
        
        
        if input_sentence == "quit" or input_sentence == "end":
            chatting = False
            break
        
        input_sentence = tokenize(input_sentence)
        x = bag_of_words(input_sentence, word_collection)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x).to(device)
        
        output = model(x)
        _, predicted = torch.max(output, dim = 1)
        tag = tags[predicted.item()]
        
        probability = torch.softmax(output, dim = 1)
        probability = probability[0][predicted.item()]
        
        unrecognized_input = "I do not understand. Please make it clear."
        
        if probability.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    random_response = random.choice(intent['responses'])
                    print(f"{bot_name}: {random_response}")
                    engine.say(random_response)
                    engine.runAndWait()
        else:
            print(unrecognized_input)
            engine.say(unrecognized_input)
            engine.runAndWait()
            
                
        
        

if __name__ == "__main__":
    main()