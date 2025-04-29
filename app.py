# from flask import Flask, request, jsonify
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch
# from flask_cors import CORS
# import logging
#
# app = Flask(__name__)
# CORS(app)
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# model_name = "pszemraj/led-large-book-summary"
# logger.info(f"Loading model {model_name}...")
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()
# logger.info(f"Model loaded on {device}")
#
# def chunk_text(text, max_tokens=14000):
#     tokens = tokenizer.encode(text, truncation=False, add_special_tokens=False)
#     chunks = []
#     for i in range(0, len(tokens), max_tokens):
#         chunk_tokens = tokens[i:i + max_tokens]
#         chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
#         chunks.append(chunk_text)
#     return chunks
#
# def summarize_text(text):
#     try:
#         inputs = tokenizer(text, return_tensors="pt", max_length=16384, truncation=True)
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         summary_ids = model.generate(
#             inputs["input_ids"],
#             attention_mask=inputs["attention_mask"],
#             max_length=300,
#             min_length=50,
#             no_repeat_ngram_size=3,
#             repetition_penalty=3.5,
#             num_beams=4,
#             early_stopping=True
#         )
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         return summary
#     except Exception as e:
#         logger.error(f"Error summarizing text: {str(e)}")
#         return f"Error: {str(e)}"
#
# @app.route("/summarize", methods=["POST"])
# def summarize():
#     try:
#         data = request.get_json()
#         if not data or "text" not in data:
#             return jsonify({"error": "No text provided"}), 400
#         input_text = data["text"]
#         if not input_text.strip():
#             return jsonify({"summary": "No summary found: Input text is empty"}), 200
#         chunks = chunk_text(input_text)
#         summaries = []
#         for chunk in chunks:
#             summary = summarize_text(chunk)
#             if not summary.startswith("Error"):
#                 summaries.append(summary)
#         if not summaries:
#             return jsonify({"summary": "No summary found: Unable to summarize any chunks"}), 200
#         combined_summary = " ".join(summaries)
#         return jsonify({"summary": combined_summary})
#     except Exception as e:
#         logger.error(f"API error: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "healthy"}), 200
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=False)

### V2 WORKS GOOD

# import os
# import time
# import tempfile
# import logging
#
# from flask import Flask, request, jsonify
# from PyPDF2 import PdfReader
#
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = "pszemraj/led-large-book-summary"  # or your model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
#
#
# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     full_text = ""
#     for i, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text:
#             full_text += text + "\n"
#         logger.info(f"Processed page {i+1}/{len(reader.pages)}")
#     return full_text
#
#
# # Chunk text into manageable parts
# def chunk_text(text, max_tokens=14000):
#     sentences = text.split(". ")
#     chunks = []
#     current_chunk = ""
#
#     for sentence in sentences:
#         if len(tokenizer.encode(current_chunk + sentence, add_special_tokens=False)) < max_tokens:
#             current_chunk += sentence + ". "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + ". "
#
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#
#     return chunks
#
#
# # Summarize a batch of text chunks
# def summarize_batch(text_list):
#     try:
#         outputs = []
#         for text in text_list:
#             inputs = tokenizer(
#                 text,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=14336,
#                 padding="max_length"
#             ).to(device)
#
#             with torch.no_grad():
#                 summary_ids = model.generate(
#                     input_ids=inputs['input_ids'],
#                     attention_mask=inputs['attention_mask'],
#                     max_length=150,
#                     min_length=20,
#                     no_repeat_ngram_size=3,
#                     repetition_penalty=3.5,
#                     num_beams=4,
#                     early_stopping=True
#                 )
#
#             summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#             outputs.append(summary)
#
#         return outputs
#
#     except Exception as e:
#         logger.error(f"Error summarizing batch: {str(e)}")
#         return [f"Error: {str(e)}"]
#
#
#
# # Summarization route
# @app.route("/summarize", methods=["POST"])
# def summarize():
#     start_time = time.time()
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
#
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "Empty file"}), 400
#
#         # Save uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#             file.save(temp_pdf.name)
#             temp_pdf_path = temp_pdf.name
#
#         # Extract text
#         input_text = extract_text_from_pdf(temp_pdf_path)
#         os.remove(temp_pdf_path)
#
#         if not input_text.strip():
#             return jsonify({"summary": "No summary found: Extracted text is empty"}), 200
#
#         # Chunk text
#         chunks = chunk_text(input_text)
#         logger.info(f"Total chunks: {len(chunks)}")
#
#         summaries = []
#         batch_size = 4
#         max_chunks = min(len(chunks), 20)  # You can increase this later
#
#         # Process in batches
#         # NEW (FIXED) version inside for loop
#         for idx, chunk in enumerate(chunks[:20]):  # Limit to first 20 chunks
#             batch_summaries = summarize_batch([chunk])  # single chunk at a time
#             summaries.extend(batch_summaries)
#             logger.info(f"Chunk {idx + 1}/{len(chunks[:20])} summarized")
#
#         if not summaries:
#             return jsonify({"summary": "No summary found: Unable to summarize any chunks"}), 200
#
#         combined_summary = " ".join(summaries)
#         logger.info(f"Total summarization took {time.time() - start_time:.2f} seconds")
#
#         return jsonify({"summary": combined_summary})
#
#     except Exception as e:
#         logger.error(f"API error: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
#
# # Run the app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


### THE SCRIPT BELLOW IS FASTER BECAUSE IT USES ONLY 5 CHUNKS - WHICH IS NOT PREFERABLE
# import os
# import time
# import tempfile
# import logging
#
# from flask import Flask, request, jsonify
# from PyPDF2 import PdfReader
#
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = "pszemraj/led-large-book-summary"  # your model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
#
#
# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     full_text = ""
#     for i, page in enumerate(reader.pages):
#         text = page.extract_text()
#         if text:
#             full_text += text + "\n"
#         logger.info(f"Processed page {i+1}/{len(reader.pages)}")
#     return full_text
#
#
# # Chunk text into manageable parts
# def chunk_text(text, max_tokens=14000):
#     sentences = text.split(". ")
#     chunks = []
#     current_chunk = ""
#
#     for sentence in sentences:
#         if len(tokenizer.encode(current_chunk + sentence, add_special_tokens=False)) < max_tokens:
#             current_chunk += sentence + ". "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + ". "
#
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#
#     return chunks
#
#
# # Summarize a batch of text chunks
# def summarize_batch(text_list):
#     try:
#         outputs = []
#         for text in text_list:
#             inputs = tokenizer(
#                 text,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=14336,
#                 padding="max_length"
#             ).to(device)
#
#             with torch.no_grad():
#                 summary_ids = model.generate(
#                     input_ids=inputs['input_ids'],
#                     attention_mask=inputs['attention_mask'],
#                     max_length=120,  # smaller = faster
#                     min_length=20,
#                     no_repeat_ngram_size=3,
#                     repetition_penalty=2.5,  # lower = faster
#                     num_beams=2,  # lower = faster
#                     early_stopping=True
#                 )
#
#             summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#             outputs.append(summary)
#
#         return outputs
#
#     except Exception as e:
#         logger.error(f"Error summarizing batch: {str(e)}")
#         return [f"Error: {str(e)}"]
#
#
# # Summarization route
# @app.route("/summarize", methods=["POST"])
# def summarize():
#     start_time = time.time()
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
#
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({"error": "Empty file"}), 400
#
#         # Save uploaded file temporarily
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
#             file.save(temp_pdf.name)
#             temp_pdf_path = temp_pdf.name
#
#         # Extract text
#         input_text = extract_text_from_pdf(temp_pdf_path)
#         os.remove(temp_pdf_path)
#
#         if not input_text.strip():
#             return jsonify({"summary": "No summary found: Extracted text is empty"}), 200
#
#         # Chunk text
#         chunks = chunk_text(input_text)
#         logger.info(f"Total chunks: {len(chunks)}")
#
#         summaries = []
#         batch_size = 1  # summarizing one chunk at a time
#         max_chunks = min(len(chunks), 5)  # FAST: summarize max 5 chunks
#
#         # Process in batches
#         for idx, chunk in enumerate(chunks[:max_chunks]):
#             batch_summaries = summarize_batch([chunk])  # single chunk
#             summaries.extend(batch_summaries)
#             logger.info(f"Chunk {idx + 1}/{max_chunks} summarized")
#
#         if not summaries:
#             return jsonify({"summary": "No summary found: Unable to summarize any chunks"}), 200
#
#         combined_summary = " ".join(summaries)
#         logger.info(f"Total summarization took {time.time() - start_time:.2f} seconds")
#
#         return jsonify({"summary": combined_summary})
#
#     except Exception as e:
#         logger.error(f"API error: {str(e)}")
#         return jsonify({"error": str(e)}), 500
#
# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify({"status": "healthy"}), 200
#
# # Run the app
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

### MODEL WORKS, 18min 31sec for 183 page book - it gets the pdf from the springboot app then creates chunks and sends to the model to summarize

import os
import time
import tempfile
import logging
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "pszemraj/led-large-book-summary"  # your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"
        logger.info(f"Processed page {i+1}/{len(reader.pages)}")
    return full_text

# Chunk text into manageable parts
def chunk_text(text, max_tokens=14000):
    sentences = text.split(". ")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(tokenizer.encode(current_chunk + sentence, add_special_tokens=False)) < max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Summarize a batch of text chunks
def summarize_batch(text_list):
    try:
        outputs = []
        for text in text_list:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=14336,
                padding="max_length"
            ).to(device)

            with torch.no_grad():
                summary_ids = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=150,  # Keep it small and efficient
                    min_length=20,
                    no_repeat_ngram_size=3,
                    repetition_penalty=3.5,
                    num_beams=4,
                    early_stopping=True
                )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            outputs.append(summary)

        return outputs
    except Exception as e:
        logger.error(f"Error summarizing batch: {str(e)}")
        return [f"Error: {str(e)}"]

# Summarization route
@app.route("/summarize", methods=["POST"])
def summarize():
    start_time = time.time()
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty file"}), 400

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            file.save(temp_pdf.name)
            temp_pdf_path = temp_pdf.name

        # Extract text
        input_text = extract_text_from_pdf(temp_pdf_path)
        os.remove(temp_pdf_path)

        if not input_text.strip():
            return jsonify({"summary": "No summary found: Extracted text is empty"}), 200

        # Chunk text
        chunks = chunk_text(input_text)
        logger.info(f"Total chunks: {len(chunks)}")

        summaries = []

        # Sequential batch processing
        batch_size = 4  # Number of chunks to summarize at once
        max_chunks = len(chunks)  # Take all chunks
        logger.info(f"Summarizing {max_chunks} chunks...")

        # Process chunks in sequential batches
        for i in range(0, max_chunks, batch_size):
            chunk_batch = chunks[i:i + batch_size]
            batch_summaries = summarize_batch(chunk_batch)
            summaries.extend(batch_summaries)

        if not summaries:
            return jsonify({"summary": "No summary found: Unable to summarize any chunks"}), 200

        combined_summary = " ".join(summaries)
        logger.info(f"Total summarization took {time.time() - start_time:.2f} seconds")

        return jsonify({"summary": combined_summary})

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

