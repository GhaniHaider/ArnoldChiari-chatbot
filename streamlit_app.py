import torch
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import fitz  # PyMuPDF

# Load the RAG model and tokenizer
model_name = "facebook/rag-token-base"
tokenizer = RagTokenizer.from_pretrained(model_name)
retriever = RagRetriever.from_pretrained(model_name, index_name="custom", passages_path="passages", index_path="index")
model = RagTokenForGeneration.from_pretrained(model_name, retriever=retriever)

# Load the PDF and extract text
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Preprocess the text into passages
def preprocess_text_into_passages(text, passage_length=200):
    words = text.split()
    passages = []
    for i in range(0, len(words), passage_length):
        passage = " ".join(words[i:i + passage_length])
        passages.append(passage)
    return passages

# Save passages and create an index
def create_retriever_index(passages, passages_path="passages", index_path="index"):
    with open(passages_path, "w") as f:
        for passage in passages:
            f.write(passage + "\n")
    retriever = RagRetriever.from_pretrained(model_name, index_name="custom", passages_path=passages_path, index_path=index_path)
    return retriever

# Generate a response using RAG
def generate_response(question):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model.generate(input_ids)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Main function to handle the chatbot
def neurosurgery_chatbot(question):
    # Extract text from the PDF
    pdf_path = "Handbook_of_Neurosurgery.pdf"
    text = extract_text_from_pdf(pdf_path)
    
    # Preprocess text into passages
    passages = preprocess_text_into_passages(text)
    
    # Create retriever index
    retriever = create_retriever_index(passages)
    
    # Generate response
    response = generate_response(question)
    
    # Check if the response is relevant
    if "my responses are limited" in response.lower():
        return "My responses are limited, you must ask the right question."
    else:
        return response

# Example usage
if __name__ == "__main__":
    question = "What is the treatment for a subdural hematoma?"
    response = neurosurgery_chatbot(question)
    print(response)
