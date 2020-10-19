from transformers import GPT2Tokenizer, GPT2Model
import torch
def calculate_sentence_embedding(input_sentence):	
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = GPT2Model.from_pretrained('gpt2', return_dict=True)
  inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
  outputs = model(**inputs)
  last_hidden_states = outputs.last_hidden_state
  sum = torch.mean(last_hidden_states, dim=1)
  return sum.detach().numpy()[0]

