from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = [
    "Patients can cancel appointments within 24 hours.",
    "Doctors charge a consultation fee of $50.",
    "The clinic is open from 9 AM to 5 PM.",
]

chunk_embedding = model.encode(chunks)

query = "can i can cel my appointment"

query_Embedding = model.encode([query])


score = cosine_similarity(query_Embedding,chunk_embedding)

bset_index = score[0].argmax()


print(chunks[bset_index])