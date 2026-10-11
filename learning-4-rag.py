from dotenv import load_dotenv
from google import genai
import math

load_dotenv()

client = genai.Client()


# -----------------------------
# 1. Our stored documents
# -----------------------------

documents = [
    "DBMS project demo is on Friday.",
    "Each team gets 8 minutes for the DBMS project demo.",
    "CN quiz is on Monday."
]


# -----------------------------
# 2. Convert text → embedding
# -----------------------------

def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )

    return response.embeddings[0].values


# -----------------------------
# 3. Compare two embeddings
# -----------------------------

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    return dot_product / (magnitude_a * magnitude_b)


# -----------------------------
# 4. Embed stored documents once
# -----------------------------

document_embeddings = []

for document in documents:
    vector = get_embedding(document)
    document_embeddings.append(vector)


# -----------------------------
# 5. Semantic retriever
# -----------------------------

def semantic_retrieve(query, top_k=2, min_score=0.70):

    query_vector = get_embedding(query)

    scored_results = []

    for i in range(len(documents)):

        score = cosine_similarity(
            query_vector,
            document_embeddings[i]
        )

        scored_results.append(
            (score, documents[i])
        )

    scored_results.sort(
        key=lambda x: x[0],
        reverse=True
    )
    filtered_results = []

    for score, document in scored_results:
        if score >= min_score:
            filtered_results.append((score, document))

    return filtered_results[:top_k]


# -----------------------------
# 6. User asks a question
# -----------------------------

query = "What is the weather in delhi"


# -----------------------------
# 7. Retrieve relevant documents
# -----------------------------

results = semantic_retrieve(query)

if not results:
    print("I don't have enough relevant information to answer that.")

for score, document in results:
    print(score, "->", document)


# Get only the document text
retrieved_docs = []

for score, document in results:
    retrieved_docs.append(document)


# -----------------------------
# 8. Augment model context
# -----------------------------

context = "\n".join(retrieved_docs)

prompt = f"""
Answer the user's question using only the provided context.

CONTEXT:
{context}

QUESTION:
{query}
"""


# -----------------------------
# 9. Generate final answer
# -----------------------------

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt
)

print("\nFINAL ANSWER:")
print(interaction.output_text)