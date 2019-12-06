import json
from pathlib import Path

WORD_EMBEDDINGS = ["word2vec", "fasttext-crawl", "fasttext-crawl-subword", "fasttext-wiki-news", "fasttext-wiki-news-subword",
                  "elmo", "bert-base", "bert-large", "bert-service-base", "bert-service-large", "glove-50", "glove-100", "glove-200", "glove-300",
                  "random-embeddings-100", "random-embeddings-1024", "random-embeddings-200", "random-embeddings-300",
                  "random-embeddings-50", "random-embeddings-768", "random-embeddings-850", "wordnet2vec"]

def generate_config_json(path=Path('.'),
                         cd='mitchell',
                         dim=1000,
                         start=0,
                         end=2,
                         wordEmbeddings=WORD_EMBEDDINGS):

    config_dict = {'cognitiveData':{}}
    for i in range(start,end+1):
        config_dict['cognitiveData'][cd+'-'+str(dim)+'-'+str(i)] = {
            "features": [
                "ALL_DIM"
            ]
            }

    config_dict['wordEmbeddings'] = wordEmbeddings
    config_dict["configFile"] = "config/setupConfig.json"

    with open(path / (cd+'-'+str(dim)+'.json'),'w') as fW:
        json.dump(config_dict,fW, indent=4, sort_keys = True)

if __name__ == "__main__":
    generate_config_json()