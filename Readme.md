# RAG QA Chatbot with Memory

A Question-Answering chatbot built using RAG (Retrieval-Augmented Generation) with conversation memory. This project uses LangChain, various LLM options, and vector stores to create an intelligent chatbot that can answer questions about Jessup Cellars winery.

## Features

- RAG-based question answering
- Conversation memory to maintain context
- Support for multiple LLM options (Groq, OpenAI)
- Vector store options (Pinecone, FAISS)
- Environment variable configuration
- Flexible embedding models (HuggingFace, OpenAI)

## Installation

#### 1.Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

#### 2.Create and activate a virtual environment:
```bash
virtualenv env
env/Scripts/activate
```

#### 3.Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

### 1.Create a `.env` file in the project root and add your API keys:

```bash
LANGCHAIN_API_KEY="your_langchain_api_key"
LANGCHAIN_PROJECT="RAG QA Chatbot with Memory"
OPENAI_API_KEY="your_openai_api_key"
GOOGLE_API_KEY="your_google_api_key"
GROQ_API_KEY="your_groq_api_key"
HUGGING_FACE_TOKEN="your_huggingface_token"
PINECONE_API_KEY="your_pinecone_api_key"
```


### For Google Colab, use secrets to store API keys:

```bash
from google.colab import userdata
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
# Add other API keys as needed
```

# LLM Options

## Default Configuration (Open Source)

The project uses Groq's Llama3-70b-8192 model by default, as it's a powerful open-source alternative:

```bash
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-70b-8192",
    temperature=0
)
```
## OpenAI Configuration (Commented Out)

OpenAI Configuration (Commented Out)

```bash
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=4,
    api_key=openai_api_key,
)
```
## Vector Store Options

### Pinecone (Default)
The project uses Pinecone as the default vector store for production use.

### FAISS (Local Alternative)
For local development or testing, uncomment the FAISS implementation in the code.

## Running the Project

- 1.Ensure all configurations are set up properly
- 2.Run the main script:

```bash
python yard.py
```
- 3.Start asking questions about Jessup Cellars.
- 4.Type 'exit' to end the session.


## Sample Usage

```bash
Enter your question: What makes Jessup Cellars wines special?
[Response will appear here]
Response time: [time in seconds]
##################################################
Enter your question:
```
## Note on LLM Choice
- This project was developed using open-source LLMs (Groq's Llama3-70b-8192) due to OpenAI API credit limitations. The code includes commented sections for OpenAI integration if you have API credits available.

## Corpus Information

- See `corpus_info.md` for detailed information about the Jessup Cellars knowledge base used in this project.