# MultiSource Explorer Chatbot

MultiSource Explorer Chatbot is a Flask-based web application that enables users to interact with a chatbot using either URLs or PDF documents. The chatbot utilizes OpenAI language models and leverages the langchain library for natural language processing and information retrieval.

## Features

- **URL Processing:** Extract information from provided URLs, create embeddings, and save to a FAISS index for future queries.
- **PDF Processing:** Extract text from uploaded PDFs, create embeddings, and save to a FAISS index for future queries.
- **Question Answering:** Ask questions to the chatbot, which retrieves relevant information from the processed data and provides answers.
- **Source Tracking:** Display sources of information retrieved for transparency.

# Output

![Animated GIF](MultiSourceExplorerChatbot.gif)

