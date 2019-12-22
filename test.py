import json

with open('batch_9_meiqw.jsonl.json','r') as f:
    lines = f.readlines()
    for line in lines[:200]:
        print(line)
        doc_json = json.loads(line)
        print(doc_json)
    #for p in data["text"].split():
        #print(p)
