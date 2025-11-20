from transformers import pipeline
import pandas as pd
import textwrap
import torch, gc, re
from retrieve import RAGRetriever

# Load FLAN-T5 (runs on CPU or GPU automatically)
def create_generator():
    try:
        # Clear cache first
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            device_map="auto",
            torch_dtype=torch.float16,
            max_length=400
        )
        return generator
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Build prompt from retrieved rows
def build_prompt(query: str, retrieved_df: pd.DataFrame, requested_items: int|None=None) -> str:
    """
    Construct a readable context for the model.
    Each product adds title, price, rating, and category info.
    """
    
    query_lower = query.lower()
    is_budget_query = any(word in query_lower for word in ['cheap', 'budget', 'affordable', 'under'])
    is_premium_query = any(word in query_lower for word in ['premium', 'high-end', 'expensive', 'over'])
    needs_high_rating = any(word in query_lower for word in ['good rating', 'high rating', 'highly rated'])
    
    # Build context-specific guidance
    guidance_parts = []
    if is_budget_query:
        guidance_parts.append("Focus on affordable options and highlight the best value for money.")
    if is_premium_query:
        guidance_parts.append("Emphasize premium features and build quality.")
    if needs_high_rating:
        guidance_parts.append("Prioritize products with higher ratings and mention the rating prominently.")
    
    specific_guidance = " ".join(guidance_parts)
    
    """
    Automatically adjust title/description length based on number of items.
    More items = shorter text, Fewer items = longer text.
    """
    actual_items = min(len(retrieved_df), 3)

    # Calculate dynamic lengths based on number of items
    if actual_items <= 2:
        # For 1-2 items: almost full length
        title_length = 120
        desc_length = 150
    elif actual_items <= 4:
        # For 3-4 items: medium length
        title_length = 80
        desc_length = 100
    else:
        # For 5+ items: shorter length
        title_length = 50
        desc_length = 60

    context_parts = []
    for i, row in retrieved_df.head(actual_items).iterrows():
        # Dynamic truncation based on calculated lengths
        title = str(row['title'])
        if len(title) > title_length:
            title = title[:title_length] + "..."
        
        description = str(row['text'])
        if len(description) > desc_length:
            description = description[:desc_length] + "..."
        
        price = row.get('price', 'N/A')
        if isinstance(price, (int, float)):
            price = f"{price} EUR"

        part = textwrap.dedent(f"""
            Product {i+1}:   
            - Title: {title}
            - Category: {row.get('categories', 'General')}
            - Price: {price}
            - Rating: {row.get('average_rating', 'N/A')}
            - Description: {description}
        """).strip()
        context_parts.append(part)
    
    context = "\n\n".join(context_parts)

    prompt = textwrap.dedent(f"""
    You are a friendly shopping expert specializing in Beauty and Electronics.

    Below is a list of products retrieved for the user’s question.
    Each product includes its name, category, price (EUR), and rating (1–5 stars).

    Follow these steps carefully:
    1. Review all products in the list.
    2. Compare their ratings and prices to find the best fits for the user’s need or budget.

    Do not invent new products or specifications; use only the information provided.

    Products:
    {context}

    Question: {query}
    Answer:
    """).strip()
    return prompt

def generate_answer(query: str, retrieved_df: pd.DataFrame, generator, requested_items: int|None=None,max_tokens=200):
    """Generate answer with memory management"""
    try:
        if retrieved_df.empty:
            return "I couldn't find any relevant products for that query."
        
        # Clear memory before generation
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        prompt = build_prompt(query, retrieved_df, requested_items)  # Reduced to 2 items
        print("\n----- PROMPT SENT TO MODEL -----\n")
        print(prompt)
        print("\n-------------------------------\n")
        
        print("Generating response...")
        response = generator(
            prompt, 
            max_tokens=max_tokens, 
            do_sample=True,
            temperature=0.1
        )
        
        # Clear memory after generation
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return response[0]['generated_text'].strip()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Memory error - clearing cache and retrying...")
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            return "The system is busy. Please try again or close other applications."
        else:
            return f"Model error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
    
if __name__ == '__main__':
    try:
        rag = RAGRetriever()
        generator = create_generator()

        print("Search Beauty & Electronics products (type 'exit' to quit).")
        while True:
            try:
                query = input("\nSearch: ").strip()
                if query.lower() in ['exit', 'quit', 'e', 'q']:
                    break
                elif not query:
                    continue
                    
                print("Retrieving products...")
                results = rag.retrieve(query, False)
                df = results["dataframe"]
                item_count = results["requested_items"]
                print(f"Retrieved {len(df)} products")
                
                print("Generating answer...")
                answer = generate_answer(query, df, generator, requested_items=item_count)
                
                print(f"\n{'='*50}")
                print(f"Query: {query}")
                print(f"Answer: {answer}")
                print(f"{'='*50}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                continue
                
    except Exception as e:
        print(f"Failed to initialize: {e}")