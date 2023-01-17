from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
inputs = tokenizer("I like to write about Conversational AI", return_tensors="pt")
res = model.generate(**inputs)
print(tokenizer.decode(res[0]))
print(tokenizer.decode(inputs['input_ids'][0]))
