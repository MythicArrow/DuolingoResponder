from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Example Duolingo-style question
question = input("First write the translation type then write the sentence")

# Tokenize input
inputs = tokenizer.encode(question, return_tensors="pt")

# Generate answer
outputs = model.generate(inputs)

# Decode output
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Answer:", answer)