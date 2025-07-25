def create_user_profile(qa_pairs: list[dict]) -> str | None:
    """Creates a descriptive string directly from a list of Q&A pairs."""
    if not qa_pairs:
        return None
    
    # Extract just the answers
    answers = [item.answer for item in qa_pairs]
    
    descriptive_answers = [
        ans for ans in answers 
        if ans.lower() not in ['yes', 'no']
    ]
    return ". ".join(descriptive_answers)

async def get_recommendations(user_profile_text: str, index, client, top_k: int = 3) -> list:
    """Gets fragrance recommendations from Pinecone."""
    if not user_profile_text:
        return []
    
    try:
        res = await client.embeddings.create(
            input=[user_profile_text], 
            model='text-embedding-3-large'
        )
        query_embedding = res.data[0].embedding
        query_results = index.query(
            vector=query_embedding, 
            top_k=top_k, 
            include_metadata=True
        )
        return query_results['matches']
    except Exception as e:
        print(f"An error occurred during embedding or querying: {e}")
        return []

async def generate_reasoning_with_llm(user_profile: str, fragrance_metadata: dict, client) -> str:
    """Uses a powerful LLM to generate deep, narrative reasoning for a recommendation."""
    parts = []
    if fragrance_metadata.get("perfume_name"): parts.append(f"Perfume Name: {fragrance_metadata['perfume_name']}")
    olfactive = fragrance_metadata.get("olfactive_profile", {})
    if olfactive.get("family"): parts.append(f"Family: {olfactive['family']}")
    semantic = fragrance_metadata.get("semantic_profile", {})
    if semantic:
        if semantic.get("gender"): parts.append(f"Gender: {semantic['gender']}")
        if semantic.get("occasion"): parts.append(f"Occasion: {semantic['occasion']}")
        if semantic.get("mood"): parts.append(f"Mood: {semantic['mood']}")
    fragrance_details = ". ".join(parts)

    prompt_messages = [
        {
            "role": "system",
            "content": '''You are a world-class Fragrance Concierge. Your task is to provide a sophisticated and persuasive reasoning for a perfume recommendation. Synthesize the user's preferences with the perfume's characteristics. Go beyond simple matching and create a short, elegant narrative. Speak directly to the user. 
            Speak directly to the user. Keep it persuasive, under 30 words, and conversational.
            At least one reasoning must refer explicitly to the user’s preferences.
            The tone should feel like a confident friend helping them make a smart choice.
            Avoid fluff or poetic language.'''
        },
        {
            "role": "user",
            "content": f"User Preference Profile: \"{user_profile}\"\n\nRecommended Fragrance Details: \"{fragrance_details}\"\n\nPlease generate the reasoning text."
        }
    ]
    try:
        response = await client.chat.completions.create(
            model='gpt-4.1',
            messages=prompt_messages,
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\nError generating reasoning: {e}")
        return "Could not generate reasoning due to an API error."
