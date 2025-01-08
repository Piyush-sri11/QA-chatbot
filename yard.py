from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
load_dotenv()

# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

groq_api_key=os.getenv('GROQ_API_KEY')
inference_api_key=os.getenv('HUGGING_FACE_TOKEN')
pinecone_api_key=os.getenv('PINECONE_API_KEY')
openai_api_key=os.getenv('OPENAI_API_KEY')

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-70b-8192",temperature=0)

# To use openai model uncomment the below code and comment the above code

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=4,
#     api_key=openai_api_key,
# )

# to use OpenAI embeddings uncomment the below code and comment the HUggingFace code
# embeddings = OpenAIEmbeddings(
#     model="text-embedding-3-large",
#     api_key=openai_api_key,
# )

embeddings=HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key, model_name="BAAI/bge-large-en-v1.5"
    )

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "qa-ndex"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="enabled",  # Defaults to "disabled"
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

#use FAISS for local storage of embeddings

# if not os.path.exists("faiss_index"):
#     ### Construct retriever ###
#     loader =PyPDFLoader("Corpus.pdf")
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(docs)
#     vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
#     vectorstore.save_local("faiss_index")

# vectorstore=FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = (
    """Given a chat history and the latest user question, which might reference context from the chat history, formulate a standalone question that can be understood without the need for the chat history. Do NOT answer the question; simply reformulate it if needed and otherwise if the question is already standalone, return it as it is., particularly focusing on the context of Jessup Cellars and the role of a Product and Sales Manager at the Jessup Cellars Wine Company.
    """
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    """You are an assistant for answering questions related to the operations, culture, history, wines, wine-making processes, and people of Jessup Cellars. Use the provided context on Jessup Cellars Corpus as your sole source of information. Act like a Product and Sales Manager from Jessup Cellars, providing clear and detailed answers.Use the sample interactions to form answer. If a question falls outside the context of the corpus, respond with: "Contact Jessup Cellars directly for more information.
    \n\n
    <Corpus>
    {context}
    </Corpus>

    Sample Interaction:

    "question": "What makes Jessup Cellars wines special?",
    "answer": "Jessup Cellars wines are carefully crafted with the help of our renowned consulting winemaker Rob Lloyd who famously crafted        
     Chardonnay for Rombauer, La Crema and Cakebread. Not only has Rob created one of the best Chardonnays in the Napa Valley with our 2022 vintage, but has also helped curate 'The Art of the Blend' with our stellar red wines."
     
    When asked about any person and their role at Jessup Cellars, respond in the following manner:
    "question": "Who is your winemaker at the winery in Napa?",
    "answer": "Bernardo Munoz\nWinemaker\n\nBIOGRAPHY\nHometown: Campeche, Mex\n\nFavorite Jessup Wine Pairing: 2010 Manny\u2019s Blend with 
     Mongolian Pork Chops from Mustards Grill \u2013 richness paired with richness\n\nAbout: Bernardo began his career in the vineyards, learning the intricacies of grape growing and how to adjust to the whims of Mother Nature. He then moved into the cellars at Jessup Cellars, bringing us his complete grape to bottle knowledge to the team. He has a keen understanding of what it takes to make a great bottle."
    """
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


import time

query=input("Enter your question: ")
while(query!="exit"):
    start=time.process_time()
    answer = conversational_rag_chain.invoke(
    {"input": query},
    config={"configurable": {"session_id": "gt9080"}},)
    print(answer["answer"])
    print("Response time :",time.process_time()-start)
    print("#"*50)
    query=input("Enter your question:")
else:
    print("Thank You! Have a nice day.")

