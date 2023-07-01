
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# load the document(s)
loader = TextLoader('./documents/Sample.txt')
documents = loader.load()

# transform (chunk) document(s) into doc objects
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
texts = text_splitter.split_documents(documents)

# text embedding to convert document objects to vectors
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# store text embeddings in vectorstore
db = Chroma.from_documents(texts, embeddings)

# retrieve text embeddings of docs
retriever = db.as_retriever(search_kwargs={"k": 2})  # k=top doc results

# query using retriever to perform semantic search
docs = retriever.get_relevant_documents("What is the capital of india?")
print(docs)
docs = retriever.get_relevant_documents("What is the currency india?")
print(docs)
