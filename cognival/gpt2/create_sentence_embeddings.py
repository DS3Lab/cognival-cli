import pandas as pd
import gpt_embeddings

inputs = pd.read_csv("./bert_sentences.txt", header=None, delimiter="\n")

list_of_outputs = []

for i in range(len(inputs)):
  print(f'Calculating embedding for sentence nr. {i}')
  list_of_outputs.append(gpt_embeddings.calculate_sentence_embedding(inputs.iloc[i]))

outputs = pd.DataFrame(list_of_outputs)

result = pd.concat([inputs, outputs], axis=1)
result.to_csv("result.txt", header=None)
print("Finished! Embeddings written to result.txt")

