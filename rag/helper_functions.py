from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import PromptTemplate
import fitz
from typing import List
from rank_bm25 import BM25Okapi
import asyncio
import random
import textwrap
import numpy as np



def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.
    将每个文档页面内容中的所有制表符('\t')替换为空格。

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents


def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.
    将输入文本换行到指定的宽度。

    Args:
        text (str): The input text to wrap.
        width (int): The width at which to wrap the text.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)




def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.
    使用OpenAI嵌入将PDF书编码为矢量存储。

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([content])

    for chunk in chunks:
        chunk.metadata['relevance_score'] = 1.0
        
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_context_per_question(question, chunks_query_retriever):
    """
    Retrieves relevant context and unique URLs for a given question using the chunks query retriever.
    使用块查询检索器检索给定问题的相关上下文和唯一url。

    Args:
        question: The question for which to retrieve context and URLs.

    Returns:
        A tuple containing:
        - A string with the concatenated content of relevant documents.
        - A list of unique URLs from the metadata of the relevant documents.
    """

    # Retrieve relevant documents for the given question
    docs = chunks_query_retriever.get_relevant_documents(question)

    # Concatenate document content
    # context = " ".join(doc.page_content for doc in docs)
    context = [doc.page_content for doc in docs]

    
    return context

class QuestionAnswerFromContext(BaseModel):
    """
    Model to generate an answer to a query based on a given context.
    模型生成基于给定上下文的查询的答案。
    
    Attributes:
        answer_based_on_content (str): The generated answer based on the context.
    """
    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")

def create_question_answer_from_context_chain(llm):

    # Initialize the ChatOpenAI model with specific parameters
    question_answer_from_context_llm = llm

    # Define the prompt template for chain-of-thought reasoning
    question_answer_prompt_template = """ 
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """

    # Create a PromptTemplate object with the specified template and input variables
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    # Create a chain by combining the prompt template and the language model
    question_answer_from_context_cot_chain = question_answer_from_context_prompt | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain



def answer_question_from_context(question, context, question_answer_from_context_chain):
    """
    Answer a question using the given context by invoking a chain of reasoning.
    通过调用推理链，使用给定的上下文回答问题。

    Args:
        question: The question to be answered.
        context: The context to be used for answering the question.

    Returns:
        A dictionary containing the answer, context, and question.
    """
    input_data = {
        "question": question,
        "context": context
    }
    print("Answering the question from the retrieved context...")

    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}


def show_context(context):
    """
    Display the contents of the provided context list.
    显示所提供的上下文列表的内容。

    Args:
        context (list): A list of context items to be displayed.

    Prints each context item in the list with a heading indicating its position.
    """
    for i, c in enumerate(context):
        print(f"Context {i+1}:")
        print(c)
        print("\n")


def read_pdf_to_string(path):
    """
    Read a PDF document from the specified path and return its content as a string.
    从指定的路径读取PDF文档，并将其内容作为字符串返回。

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: The concatenated text content of all pages in the PDF document.

    The function uses the 'fitz' library (PyMuPDF) to open the PDF document, iterate over each page,
    extract the text content from each page, and append it to a single string.
    """
    # Open the PDF document located at the specified path
    doc = fitz.open(path)
    content = ""
    # Iterate over each page in the document
    for page_num in range(len(doc)):
        # Get the current page
        page = doc[page_num]
        # Extract the text content from the current page and append it to the content string
        content += page.get_text()
    return content



def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5) -> List[str]:
    """
    Perform BM25 retrieval and return the top k cleaned text chunks.
    执行BM25检索并返回前k个清理过的文本块。

    Args:
    bm25 (BM25Okapi): Pre-computed BM25 index.
    cleaned_texts (List[str]): List of cleaned text chunks corresponding to the BM25 index.
    query (str): The query string.
    k (int): The number of text chunks to retrieve.

    Returns:
    List[str]: The top k cleaned text chunks based on BM25 scores.
    """
    # Tokenize the query
    query_tokens = query.split()

    # Get BM25 scores for the query
    bm25_scores = bm25.get_scores(query_tokens)

    # Get the indices of the top k scores
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]

    # Retrieve the top k cleaned text chunks
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]

    return top_k_texts



async def exponential_backoff(attempt):
    """
    Implements exponential backoff with a jitter.
    用抖动实现指数回退。
    
    Args:
        attempt: The current retry attempt number.
        
    Waits for a period of time before retrying the operation.
    The wait time is calculated as (2^attempt) + a random fraction of a second.
    """
    # Calculate the wait time with exponential backoff and jitter
    wait_time = (2 ** attempt) + random.uniform(0, 1)
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")
    
    # Asynchronously sleep for the calculated wait time
    await asyncio.sleep(wait_time)

async def retry_with_exponential_backoff(coroutine, max_retries=5):
    """
    在遇到RateLimitError时使用指数回退重试协程。

    参数：
        coroutine: 要执行的协程。
        max_retries: 最大重试尝试次数。

    返回：
        如果成功，返回协程的结果。

    抛出：
        如果所有重试尝试都失败，则抛出最后遇到异常。

    """
    for attempt in range(max_retries):
        try:
            # Attempt to execute the coroutine
            return await coroutine
        except RateLimitError as e:
            # If the last attempt also fails, raise the exception
            if attempt == max_retries - 1:
                raise e
            
            # Wait for an exponential backoff period before retrying
            await exponential_backoff(attempt)
    
    # If max retries are reached without success, raise an exception
    raise Exception("Max retries reached")
