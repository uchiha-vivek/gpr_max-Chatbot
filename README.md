
# Approach Google Summer of Code 2025 -  gprMax


Project :  AI Chatbot for support

This project aims to improve the user experience of gprMax, an open-source software for simulating electromagnetic wave propagation by modelling Ground Penetrating Radar (GPR) and electromagnetic wave propagation, through the development of an AI chatbot and assistant. The purpose of the AI chatbot is to answer and troubleshoot any questions that the gprMax users may have, while the AI assistant is capable of turning natural language into an input file in the required gprMax format. We leverage pretrained LLM models from OpenAI, the LangChain framework, RAG, and fine-tuning techniques, drawing on the existing gprMax documentation and years worth of discussion in the Google groups and the GitHub issue tracker for data. The ultimate goal is to deliver and deploy an AI chatbot and assistant to enhance the accessibility of gprMax, as well as save both the users and the development team valuable time by automating the troubleshooting process.


**contributor :** Vivek Sharma

**Mentors:** Iraklis Giannakis, Antonis Giannopoulos  





# What i have made :

**Upload PDF**

1. User selects and upload a DOCUMENT (PDF in this case)
2. The file is locally saved to document/pdfs


**Extraction and Chunking of Test**

1. PDFPlumberLoader extracts text from the PDF.
2. The text is split into overlapping chunks (1000 characters, 200 overlap) using RecursiveCharacterTextSplitter . It can be fine-tuned accordingly .


**Indexing and Vectorizing**

1. The extracted text chunks are converted into vector embeddings using OllamaEmbeddings.
2. These embeddings are stored in InMemoryVectorStore (In this case i have used InMemoryVectorStore)


**Query Processing & Answer Generation**

1. When user queries anything , the text is first converted to vector . 
2. Similarity search finds the most relevant document chunks.
3. Here i have used open-source Large Language Models like 

- deepseek-r1:1.5b
- llama3.2:3b
- llama3.2:1b



# IMPORTANT NOTE
- The main aim of this repository it to show the scope and use case on how the open-source llms can be used as chatbot . 
- I have tried to implement that how Open-source LLM's and chroma DB vector can be used for storing embeddings .
- The current code is not modular , the modular approach would be followed in actual codebase . 
similar to : https://github.com/eddieleejw/gprmax_chatbot
- I have understood that what necessary changes needs to be made in actual codebase 









