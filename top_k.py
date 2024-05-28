import streamlit as st
from openai import AzureOpenAI
import faiss
import numpy as np
import pandas as pd

#エンべディングされたデータの読み込み
df = pd.read_csv("emb_data.csv")

client = AzureOpenAI(
    api_key="0e2de9aa69fc460a8962184c8b783877",  
    api_version="2024-02-01",
    azure_endpoint="https://oai-sci24tf-pra-eus2-001.openai.azure.com/"
)

def make_index(data):
    """
    インデックスの作成
    """
    # インデックスの生成
    index = faiss.IndexFlatIP(1536)
    # 対象テキストの追加
    tmp = [i for i in data["answer_emb"].map(lambda x: [float(j) for j in x[1:-1].split(', ')]).values]
    index.add(np.array(tmp).astype('float32'))
    return index

index = make_index(df)

# "question_emb"列の値を浮動小数点数のリストに変換
df["question_emb"] = df["question_emb"].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
question_emb_np = np.stack(df["question_emb"].values).astype('float32')

retreval_num = 3    #候補件数
N = 838             #サーチしたい質問件数
true_count=0        #正しくサーチできた数

for i in range(N):
    _, I = index.search(np.array([question_emb_np[i]]), retreval_num)
    if i in I[0]:
        true_count += 1

top_k_score = true_count/N

print("---------------")
print("top_k_score:"+str(top_k_score))
print("---------------")
