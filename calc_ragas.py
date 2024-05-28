import streamlit as st
from openai import AzureOpenAI
import faiss
import numpy as np
import pandas as pd
import ast
import os
from ragas import evaluate  # 修正箇所
from ragas.metrics import answer_relevancy, faithfulness

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    answer_correctness,
)
from datasets import load_dataset, Dataset


metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness,
]

azure_model = AzureChatOpenAI(
    openai_api_version=os.environ.get('OPENAI_API_VERSION'),
    azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
    azure_deployment=os.environ.get('model'),
    model=os.environ.get('model'),
    validate_base_url=False
)

# init the embeddings for answer_relevancy, answer_correctness and answer_similarity
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version=os.environ.get('OPENAI_API_VERSION'),
    azure_endpoint=os.environ.get('AZURE_ENDPOINT'),
    azure_deployment=os.environ.get('text_emb'),
    model=os.environ.get('text_emb')
)

#環境変数設定
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_VERSION'] = "2024-02-01"
os.environ['AZURE_ENDPOINT'] =os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['model'] = "gpt-35-turbo"
os.environ['text_emb'] = "text-embedding-ada-002"

client = AzureOpenAI(
    api_key="6866b18d7e19458eb763843f04681865", 
    api_version="2024-02-01",
    azure_endpoint="https://oai-sci24tf-pra-eus2-001.openai.azure.com/"
)

# Load the data
df = pd.read_csv("emb_data.csv")
questions = df["question"]

def make_index(data):
    """
    Create FAISS index.
    """
    index = faiss.IndexFlatIP(1536)
    # Convert the answer embeddings from string representation to list of floats
    tmp = [ast.literal_eval(i) if isinstance(i, str) else i for i in data["answer_emb"].values]
    index.add(np.array(tmp).astype('float32'))
    return index

index = make_index(df)

df["question_emb"] = df["question_emb"].apply(lambda x: [float(i) for i in x[1:-1].split(', ')])
question_emb_np = np.stack(df["question_emb"].values).astype('float32')

# contextsとanswersの初期化
contexts = []
answers = []
ground_truth = []
questions_list = []

# 質問数
for question_num in range(2):
    # 各質問に関連する情報を格納するリスト
    question_contexts = []
    
    # 情報の取得数
    retrieval_num = 3
    
    # 質問に対する検索
    _, I = index.search(np.array([question_emb_np[question_num]]), retrieval_num)

    # システムプロンプトと質問を表示
    system_prompt = "以下の質問に答えてください。参考になる情報をいくつか提示しますので、それを踏まえて一言で回答を作成してください。\n"
    messages = [{"role": "system", "content": system_prompt}] + [{"role": "system", "content": df["question"][question_num]}]
    
    # 情報の取得と表示
    for i in range(retrieval_num):
        # ソースデータから回答を取得
        source = df.loc[I[0][i]]
        ans = ast.literal_eval(source.answer)["text"]
        
        # 取得した回答を表示
        messages.append({'role': 'user', 'content': f'情報{i}:{ans}\n'})
        
        # 取得した回答をリストに格納
        question_contexts.append(ans)
    
    # 質問に対する回答のリストをcontextsに格納
    contexts.append(question_contexts)
    response = client.chat.completions.create(model="gpt-35-turbo", 
                                              temperature=0.3,
                                              max_tokens=3000,
                                              messages=messages)

    answers.append(response.choices[0].message.content)
    ground_truth.append(ast.literal_eval(df["answer"][question_num])["text"])
    questions_list.append(df["question"][question_num])

# データセットの準備
ds = Dataset.from_dict(
    {
        "question": questions_list,
        "ground_truth": ground_truth,
        "answer": answers,
        "contexts": contexts,
    }
)

result = evaluate(
    ds, metrics=metrics, llm=azure_model, embeddings=azure_embeddings
)
print(result)

# 結果の表示
# print(contexts)
# print(answers)
# print(ground_truth)
