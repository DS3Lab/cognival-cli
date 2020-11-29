import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
import torch

if __name__=="__main__":
  input_df = pd.read_csv("./sentence_vocabulary.txt", header=None, delimiter="\n")
  print(f'Finished reading {len(input_df)} input sentences')
  list_of_outputs = []

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = GPT2Model.from_pretrained('gpt2', return_dict=True)
  for i in range(len(input_df)):
    print(f'Calculating embedding for sentence nr. {i}')
  
    inputs = tokenizer(input_df.iloc[i][0], return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    sum = torch.mean(last_hidden_states, dim=1)
  
    list_of_outputs.append(sum.detach().numpy()[0])

  outputs = pd.DataFrame(list_of_outputs)

  result = pd.concat([input_df, outputs], axis=1)
  result.to_csv("result.txt", header=None, sep=" ", index_col=None)
  print("Finished! Embeddings written to result.txt")
