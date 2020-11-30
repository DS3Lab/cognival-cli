import pandas as pd
import importlib
import torch
import sys
import transformers
if __name__=="__main__":
  
  model = sys.argv[1] #Name of the huggingface model
  vocab_file = sys.argv[2] #Name of the relevant huggingface vocab file 
  tokenizer_name = model + "Tokenizer"
  model_name = model + "Model"
  
  exec("from transformers import %s" % tokenizer_name)
  exec("from transformers import %s" % model_name)
  exec("tokenizer = %s.from_pretrained(vocab_file)" % tokenizer_name)
  exec("model = %s.from_pretrained(vocab_file, return_dict=True)" % model_name)
 
  input_df = pd.read_csv("./sentence_vocabulary.txt", header=None, delimiter="\n")
  print(f'Finished reading {len(input_df)} input sentences')
  list_of_outputs = []

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
