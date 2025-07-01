import os
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# CONFIG
xml_folder = "C:/Users/chann/Downloads/medquad_xml"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
question_list = []
answer_list = []

# PARSE XML SAFELY
for filename in os.listdir(xml_folder):
    if filename.endswith(".xml"):
        tree = ET.parse(os.path.join(xml_folder, filename))
        root = tree.getroot()
        for qa in root.findall(".//QAPair"):
            question_elem = qa.find("Question")
            answer_elem = qa.find("Answer")

            # Check if elements and their text exist
            if question_elem is not None and answer_elem is not None:
                question_text = question_elem.text.strip() if question_elem.text else ""
                answer_text = answer_elem.text.strip() if answer_elem.text else ""

                # Only add non-empty Q&A pairs
                if question_text and answer_text:
                    question_list.append(question_text)
                    answer_list.append(answer_text)

print(f"Parsed {len(question_list)} valid Q&A pairs.")

# EMBED QUESTIONS
embeddings = embedding_model.encode(question_list, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# FAISS INDEX
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# SAVE INDEX AND METADATA
faiss.write_index(index, "medquad_index.faiss")
with open("medquad_answers.pkl", "wb") as f:
    pickle.dump({"questions": question_list, "answers": answer_list}, f)

print("FAISS index and answer metadata saved.")
