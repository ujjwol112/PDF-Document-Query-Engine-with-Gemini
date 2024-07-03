# PDF-Document-Query-Engine-with-Gemini

URL: https://pdf-document-query-engine-with-gemini-crezkubefyjvewywgy4yv9.streamlit.app/

## 1. Overview

PDF-Document-Query-Engine-with-Gemini is a Streamlit-based application that allows users to upload PDF documents, extract text from them, and interact with the content through a conversational interface powered by Google Gemini's generative AI. The app provides a feature to query the content of the PDFs, and it can also collect user contact information for follow-up communication.

## 2. Features

- **PDF Text Extraction**: Upload multiple PDF files and extract text from all pages.
- **Text Chunking**: Split the extracted text into manageable chunks for better processing.
- **Embedding and Vector Store**: Use Google Gemini embeddings to convert text chunks into vectors and store them using FAISS.
- **Conversational Interface**: Query the content of the PDFs through a conversational interface using Google Gemini's generative AI.
- **User Information Collection**: Collect user contact information when prompted and store it in a CSV file.

## 3. Requirements

- Python 3.7 or higher
- Streamlit
- PyPDF2
- pandas
- PyDrive
- dotenv
- google-generativeai
- langchain
- langchain-google-genai
- langchain-community
- faiss-cpu


## 4. Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Use the sidebar to upload your PDF files.

3. Click the "Submit & Process" button to process the PDFs.

4. Ask questions in the text input box provided in the main area of the app.

5. If you prompt the chatbot with "call me," it will ask for your contact information.

## 5. Detailed Function Descriptions

### `get_pdf_text(pdf_docs)`

Extracts text from the uploaded PDF documents.

- **Parameters**: 
  - `pdf_docs`: List of uploaded PDF files.
- **Returns**: 
  - Concatenated text from all PDF pages.

### `get_text_chunks(text)`

Splits the extracted text into smaller chunks for better processing.

- **Parameters**: 
  - `text`: The extracted text from PDFs.
- **Returns**: 
  - List of text chunks.

### `get_vector_store(text_chunks)`

Converts text chunks into embeddings using Google Gemini and stores them in a FAISS index.

- **Parameters**: 
  - `text_chunks`: List of text chunks.
- **Returns**: 
  - None.

### `get_conversational_chain()`

Creates a conversational chain for querying the text chunks.

- **Parameters**: 
  - None.
- **Returns**: 
  - A Langchain QA chain configured with Google Gemini's generative model.

### `store_user_info(name, phone, email)`

Stores user contact information in a CSV file.

- **Parameters**: 
  - `name`: User's name.
  - `phone`: User's phone number.
  - `email`: User's email address.
- **Returns**: 
  - Path to the CSV file.

### `user_input(user_question)`

Processes user input and either collects contact information or queries the PDF content.

- **Parameters**: 
  - `user_question`: User's question or command.
- **Returns**: 
  - None.

### `main()`

Main function to run the Streamlit app.

- **Parameters**: 
  - None.
- **Returns**: 
  - None.

## 6. Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## 7. Contact

For any questions or suggestions, please contact Ujjwol Poudel at ujjwol.uj12@gmail.com.

