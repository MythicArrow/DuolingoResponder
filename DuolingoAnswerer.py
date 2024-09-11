from transformers import T5ForConditionalGeneration, T5Tokenizer

# Prompt user to select the model type
model_type = input("Enter the model type (e.g., 't5-small', 't5-base', 't5-large'): ")

# Load pre-trained T5 model and tokenizer based on user input
model = T5ForConditionalGeneration.from_pretrained(model_type)
tokenizer = T5Tokenizer.from_pretrained(model_type)

# Example Duolingo-style question
question = input("First write the translation type then write the sentence")

# Tokenize input
inputs = tokenizer.encode(question, return_tensors="pt")

# Generate answer
outputs = model.generate(inputs)

# Decode output
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Answer:", answer)