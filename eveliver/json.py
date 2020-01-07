import json


def load_jsonl(file_name):
    with open(file_name) as f:
        for line in f:
            yield json.loads(line.strip())
