from sentence_transformers import SentenceTransformer
from pathlib import Path
from main import ask_claude
import numpy as np

txt_files = Path("text_files").glob("*.txt")

model = SentenceTransformer('all-MiniLM-L6-v2')
#all_embeddings = []     deprecated variable for iteration with whole files being context instead of smaller chunking
#print(embedding, type(embedding), len(embedding))
print(txt_files)

# embeds an entire file (chunks may be too big to feed into claude)
"""
for i, j in enumerate(txt_files):
    text = j.read_text(encoding="utf-8")
    all_embeddings.append({"text": text, "embedding": model.encode(text)})
"""

# instead create micro chunks between files with overlap of 50 words to feed prev context just in case (window size ~150 words)
all_chunks = []
for i, j in enumerate(txt_files):
    text = j.read_text(encoding="utf-8").split() # need words
    window = 150 # word window for chunks
    ptr = 0
    while ptr < len(text) - window: # - window accounts for loop boundaries
        window_chunk = " ".join(text[ptr:ptr+window])
        all_chunks.append({"text": window_chunk, "embedding": model.encode(window_chunk)})        
        ptr += window
        ptr -= 50 # for overlap
    trailing_chunk = " ".join(text[ptr:])
    all_chunks.append({"text": trailing_chunk, "embedding": model.encode(trailing_chunk)}) # catch possible ends
#print(all_embeddings)

def cosine_similarity(vector_a, vector_b):
    dot = np.dot(vector_a, vector_b)
    similarity_score = dot / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    return similarity_score

def question_text_mapping(question_embedding: list[float], text_embeddings: list[dict[str, str | list[float]]]):
    tracked_text  = None
    highest = -1 # the absolute maximum opposite semantic meaning (since vectors are normalized)
    for chunk in text_embeddings:
        relation = cosine_similarity(question_embedding, chunk["embedding"])
        if relation > highest:
            highest = relation
            tracked_text = chunk["text"]
    #print("HIGHEST: " + str(highest))
    # averaged out weight of unrelated question against weight of related question for 0.3 as a correlating factor
    return tracked_text if highest > 0.3 else "No correlation"

# keeping question and embedding separate as ask_claude() expects a list of dicts
question = input()
question_embedding = model.encode(question)
conversation = []
chunk = question_text_mapping(question_embedding, all_chunks)
prompt_str = "Using the prior context, answer the following:"

if chunk != "No correlation":
    conversation.append({"role": "user", "content": str(chunk) + " " + prompt_str + question})
    output = ask_claude(conversation)
    print(conversation[-1]["content"]) # claude's response 
else:
    print("Document unable to be retrieved for context")
#print(question_text_mapping(question, all_embeddings))