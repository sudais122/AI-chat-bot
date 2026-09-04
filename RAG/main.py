from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("clinic_policy.txt") as file:
    text = file.read()

chunks = text.split("\n\n")

chunks_embedding = model.encode(chunks)

query = input("Ask anything: ")

query_embedding = model.encode([query])

similarity = cosine_similarity(query_embedding, chunks_embedding)

best_index = similarity[0].argmax()

print("\nBest chunk:")
print(chunks[best_index])