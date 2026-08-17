from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


texts = [
    "Employees are entitled to 24 paid vacation days per calendar year.",
    "How many annual leave days do employees get?",
    "Employees can work remotely up to three days per week."
]


embeddings = model.encode(texts)


similarity_ab = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)[0][0]


similarity_ac = cosine_similarity(
    [embeddings[0]],
    [embeddings[2]]
)[0][0]


print("Similarity between A and B:")
print(similarity_ab)


print("\nSimilarity between A and C:")
print(similarity_ac)