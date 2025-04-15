import os
#from summary import Summary
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.chains.question_answering import load_qa_chain

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
load_dotenv()



class Chatbot:
    def __init__(self):
        
        #data
        self.chat_history = []
        
        #open ai key
        self.openai_key = os.getenv('OPEN_AI_KEY')  
        
        #embeddings
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_key)
        
        #vector database
        document_chunks = self.chunk_data()
        if not document_chunks:
            raise ValueError("No document chunks available for creating the FAISS index.")

        self.vector = FAISS.from_documents(documents=document_chunks, embedding=self.embeddings)
        
        #-------------------------------------------------------------------
        #vector database (for the tascript)
        document_chunks = self.chunk_data_academic()
        if not document_chunks:
            raise ValueError("No document chunks available for creating the FAISS index.")

        self.vector_academic = FAISS.from_documents(documents=document_chunks, embedding=self.embeddings)
        #-------------------------------------------------------------------
        
        #llm
        self.llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=self.openai_key, max_tokens=4096)
        
        #output paerse 
        self._parser = StrOutputParser()
        
        
        #use similarity search to find the most related chunks
        self.retriever = self.vector.as_retriever(search_kwargs={"k": 2, "score_threshold": 0.5})
        
        #use similarity search to find the most related chunks (for the tascript)
        self.retriever_academic = self.vector_academic.as_retriever(search_kwargs={"k": 2, "score_threshold": 0.5})
        
        
        
        
        #Memory 
        instruction_to_system = """
        You are a smart and resourceful assistant working for Sultan Qaboos University (SQU). 
        Your role is to assist students, employees, staff, and the general community with their queries. 
        You analyze the data provided to you to ensure accurate and helpful responses. 

        When given a chat history and the latest user question, your task is to reformulate the question 
        into a standalone query that is clear and understandable even without the chat history. 
        Focus on extracting the intent and details from the chat to create an effective standalone question.

        Your responses must be concise, formal, and structured to support academic, professional, 
        or general inquiries.
         """
        question_maker_prompt = ChatPromptTemplate.from_messages([
            ("system", instruction_to_system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        self.question_maker_chain = question_maker_prompt | self.llm | self._parser
        
        
        #prompt 
        qa_system_prompt_student = """
        You are a specialized academic advisor for Sultan Qaboos University (SQU) students. 
            Provide detailed, supportive, and informative responses tailored to student queries using the provided context. 
            Focus on academic guidance, course information, campus resources, and student life at SQU. 
            Be friendly, encouraging, and precise in your tone.
        {context}
        """
        qa_system_prompt_employee = """
        You are a professional advisor for Sultan Qaboos University (SQU) employees. 
            Address queries related to HR policies, professional development, work-life balance, 
            and administrative support with a solution-oriented and formal tone. 
            Ensure that your responses reflect professionalism and clarity.
            your answers must be from the provided context
        {context}
        """
        qa_system_prompt_general = """
        You are a versatile assistant for Sultan Qaboos University (SQU), capable of assisting 
            students, employees, staff, and the general community. Your tone should adapt to the user's 
            needs—encouraging for students, professional for employees, and neutral for general inquiries. 
            Provide clear, concise, and accurate responses to a wide range of questions.
            your answers must be from the provided context
        {context}
        """
        
        # in this prompt mention that u will be  a clever role of an advisor where u will be given the transcript of the student along with the staudy pkan the the description of the courde , and what u have to do is that u chaeck and analyze everything and answe  the student questions . 
        # u will find theree things on the context :
        # 1. the student transcript 
        # 2. the study plan for his spechilaization (Computer science ) where there are 3 minor specilaizations which are [Artificial intelligence and data science , software engineering , Cyber security ]
        # 3. the description of the courss (outline od some of them)
        
        # u as an advisor u should anallze everything and a ct as an advisor to the student 
        # the student migjt ask questions like his status and how many semesters he is done with , his current gpa , the courses he got A in or whch grade , what couurses he should pay more attention to then u have to mention the courses where he got bad marks on . 
        # u should followup correctly with the study plan he has , u should km=now what courses he gor done with and what courses he didn't , u shoud advise them what courses to take nex semester . 
        # take care that u can not suggest a course while the pre reqisit is not taken by the user yet 
        # u can not exceed 18 credits suggestion of courses 
        # u can not go under 9 total credits 
        
        
        qa_system_prompt_academic =  """
            You are a clever academic advisor for Sultan Qaboos University (SQU) students. 
            Your role is to analyze the student's transcript, study plan, and course descriptions to provide tailored advice. 
            You will find three things in the context:
            1. The student's transcript.
            2. The study plan for their specialization in Computer Science, which includes three minor specializations: 
            [Artificial Intelligence and Data Science, Software Engineering, Cyber Security].
            3. The description of the courses (outline of some of them).

            As an advisor, you should:
            - Analyze the student's academic status, including completed semesters, current GPA, and grades.
            - Identify courses where the student excelled or struggled.
            - Advise on courses to take next semester, ensuring prerequisites are met.
            - Suggest a course load between 9 and 18 credits.
            - Ensure the advice aligns with the student's study plan and specialization.

            {context}
            """
        
        
        
        qa_prompt_student = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt_student),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        qa_prompt_employee = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt_employee),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        qa_prompt_general = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt_general),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        qa_prompt_academic = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt_academic),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        
        
        #retriever chain
        self.retriever_chain = RunnablePassthrough.assign(
            context = self.contextualized_question | self.retriever 
        )
        
        #retriever chain (for the tascript)
        self.retriever_chain_academic = RunnablePassthrough.assign(
            context = self.contextualized_question | self.retriever_academic 
        )
        
        #retrieval augmented generation chain
        self.rag_chain_student = (
            self.retriever_chain
            | qa_prompt_student
            | self.llm
            | self._parser
        )
        
        self.rag_chain_employee = (
            self.retriever_chain
            | qa_prompt_employee
            | self.llm
            | self._parser
        )
        self.rag_chain_general = (
            self.retriever_chain
            | qa_prompt_general
            | self.llm
            | self._parser
        )
        self.rag_chain_academic = (
            self.retriever_chain_academic
            | qa_prompt_academic
            | self.llm
            | self._parser
        )
        
        
   
        
    #define function that loads the pdf files in the pdf_files directory and split them into chunks ,remove self
    #directory_path = r'..\..\raw_data'
    def load_pdfs(self, directory_path = r'..\raw_data'):
        loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents
    
    
    def load_pdfs_academic(self, directory_path = r'..\cs_data'):
        loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents.")
        return documents

    
    
    def chunk_data(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        # Load documents from PDFs
        pdf_docs = self.load_pdfs()  # No need to pass the path here, default is used
        
        if not pdf_docs:
            print("No documents loaded. Please check your data sources.")
            return []  # Return an empty list if no documents are loaded
        
        chunks = text_splitter.split_documents(pdf_docs)
        return chunks
    
    def chunk_data_academic(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pdf_docs = self.load_pdfs_academic()
        chunks = text_splitter.split_documents(pdf_docs)
        return chunks
    
    
    def contextualized_question(self ,input:dict):
        if input.get("chat_history"):
            return self.question_maker_chain
        else:
            return input.get("question")
        
    
    def answer_query(self, query, role, transcript=None):
        print("we are inside chatbot.py")
        question = query
        if transcript:
            response = self.rag_chain_academic.invoke({"question": question, "chat_history": self.chat_history})
        elif role == "student":
            response = self.rag_chain_student.invoke({"question": question, "chat_history": self.chat_history})
        elif role == "employee":
            response = self.rag_chain_employee.invoke({"question": question, "chat_history": self.chat_history})
        else:
            response = self.rag_chain_general.invoke({"question": question, "chat_history": self.chat_history})
        print("Chatbot: ", response)
        self.chat_history.extend([HumanMessage(content=question), AIMessage(content=response)])    
        return response

