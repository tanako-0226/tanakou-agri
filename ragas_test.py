

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas import evaluate
import os
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
)
from datasets import load_dataset, Dataset


amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
]

#環境変数設定
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_VERSION'] = "2024-02-01"
os.environ['AZURE_ENDPOINT'] =os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ['model'] = "gpt-35-turbo"
os.environ['text_emb'] = "text-embedding-ada-002"


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

# Remove sensitive instances
amnesty_qa_df = amnesty_qa['eval'].to_pandas()
amnesty_qa_df.drop(18, inplace=True)
amnesty_qa_filtered = Dataset.from_pandas(amnesty_qa_df)

result = evaluate(
    amnesty_qa_filtered, metrics=metrics, llm=azure_model, embeddings=azure_embeddings
)
print(result)