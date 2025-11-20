import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.model_manager import ModelManager
import textwrap
from src.retrieve import RAGRetriever
  
def create_mistral_model():
    return ModelManager.get_model_and_tokenizer()

def build_prompt(query: str, retrieved_df, max_items=3):

    """
    Automatically adjust title/description length based on number of items.
    More items = shorter text, Fewer items = longer text.
    """
    actual_items = max(len(retrieved_df), max_items or 3)

    # Calculate dynamic lengths based on number of items
    if actual_items <= 2:
        # For 1-2 items: almost full length
        title_length = 120
        desc_length = 190
    elif actual_items <= 4:
        # For 3-4 items: medium length
        title_length = 80
        desc_length = 150
    else:
        # For 5+ items: shorter length
        title_length = 50
        desc_length = 100

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
            - Price: {price} 
            - Rating: {row.get('average_rating', 'N/A')}
            - Description: {description}
        """).strip()
        context_parts.append(part)
    
    context = "\n\n".join(context_parts)

    prompt = textwrap.dedent(f"""
    Each product includes its title, price, and rating.

    Your job:
    • Read all products carefully.
    • Recommend the exact number of products requested by the user. If the user asks for "top X", provide X results.
    • Mention each product name exactly as given.
    • Include their prices and ratings.
    • Explain briefly why each product is a good choice.
    • If fewer products than requested are available, show all available options.

    Write a clear, complete answer. Do NOT answer in one sentence.
    Do NOT be brief. Provide full recommendations.

    Products:
    {context}

    User question: {query}

    Answer:
    """).strip()

    return prompt

def generate_answer(query: str, retrieved_df, model, tokenizer, max_items, max_new_tokens=500):

    if retrieved_df.empty:
        return "I couldn't find any relevant products for that question."
    
    prompt = build_prompt(query, retrieved_df, max_items)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,         
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Answer:" in decoded:
        answer = decoded.split("Answer:", 1)[1].strip()
    else:
        answer = decoded.strip()

    return answer

def main():
    model, tokenizer = create_mistral_model()
    if model and tokenizer:
        try:
            rag = RAGRetriever()
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
                    answer = generate_answer(query, df, model, tokenizer, item_count)
                    
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

    else:
        print("Error loading model/tokenizer!!")

if __name__ == "__main__":
    main()