import json


def read_data(filepath):
    """
    キャッシュをすることでリロードするたびにロードされるのを防ぐ関数
    4000行程度なのでそもそも重くないかもしれない
    """
    data = json.load(filepath)
    print(data[0])
    return data

data = read_data("/wikipedia-human-retrieval-ja")

print(data)