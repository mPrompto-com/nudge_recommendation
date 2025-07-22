# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import Pinecone
# Note: No more 'utils' import
from src.logic import create_user_profile, get_recommendations, generate_reasoning_with_llm

load_dotenv()

app = FastAPI(
    title="Fragrance Recommendation API",
    description="An API to get personalized fragrance recommendations."
)

# --- Globals & Startup ---
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pinecone_client.Index("fragrance-recommendations-openai")
# Note: The global 'qa_data' is gone!

# --- Pydantic Models ---
class QAPair(BaseModel):
    question: str
    answer: str

class RecommendationRequest(BaseModel):
    fingerprint: str
    qa_pairs: list[QAPair]

class Recommendation(BaseModel):
    rank: int
    perfume_name: str
    handle: str | None
    similarity_score: float
    reasoning: str

class RecommendationResponse(BaseModel):
    fingerprint: str
    recommendations: list[Recommendation]

# --- API Endpoint ---
@app.post("/generate-recommendations", response_model=RecommendationResponse)
async def generate_recommendations_endpoint(request: RecommendationRequest):
    """
    Receives a user's fingerprint and their Q&A pairs, then returns recommendations.
    """
    print(f"Received request for fingerprint: {request.fingerprint}")
    
    # The qa_pairs are now passed directly from the request
    profile = create_user_profile(request.qa_pairs)
    if not profile:
        raise HTTPException(status_code=400, detail="Q&A pairs were empty or invalid.")

    matches = get_recommendations(profile, openai_client, pinecone_index, top_k=3)
    if not matches:
        raise HTTPException(status_code=404, detail="No recommendations could be generated for this profile.")

    output_data = {"fingerprint": request.fingerprint, "recommendations": []}
    for i, match in enumerate(matches):
        metadata = match['metadata']
        reasoning = await generate_reasoning_with_llm(profile, metadata, openai_client)
        
        output_data["recommendations"].append({
            "rank": i + 1,
            "perfume_name": metadata.get('perfume_name'),
            "handle": metadata.get("handle"),
            "similarity_score": match['score'],
            "reasoning": reasoning
        })
        
    return output_data

@app.get("/", summary="API Health Check")
def read_root():
    return {"status": "ok", "message": "Fragrance Recommendation API is running."}