from transformers import pipeline
import torch

classifier = pipeline('sentiment-analysis', device=0)  # Use GPU 0

user_input = input("Give me a sentiment to check: ")
while user_input != "stop":
    result = classifier(user_input)
    print(result)
    user_input = input("Provide another sentiment! Or enter \"stop\": ")


