def test_querys(rag):
    price_queries = [
        "cheap skincare products under 15 euros, recomend 10 products",
        "find expensive laptops over 1000€ recomend 20",
        "budget smartphones between 200 and 400 euros",
        "affordable makeup under 20€ with good ratings",
        "premium cameras over 800 euros",
        "mid-range headphones 50-150 euros",
        "gaming laptops under 1500€",
        "beauty products between 10 and 30 dollars"
    ]

    domain_queries = [
        # Beauty domain
        "best anti-aging creams for sensitive skin",
        "hair serums for frizzy hair with argan oil",
        "facial cleansers for acne prone skin",
        "vitamin C serums under 25 euros",
        "Korean beauty products for glowing skin",
        
        # Electronics domain
        "wireless bluetooth headphones with noise cancellation",
        "gaming processors for PC builds",
        "4K cameras for YouTube videos",
        "smart home devices under 50 euros",
        "laptop accessories for students"
    ]

    complex_queries = [
        "highly rated skincare products over 4 stars under 30 euros",
        "cheap electronics with good ratings for students",
        "premium beauty products with 4.5+ rating over 50€",
        "budget laptops under 500€ with at least 4 star rating",
        "expensive cameras over 1000€ with high user ratings"
    ]

    specific_queries = [
        # Beauty specific
        "eye creams for dark circles and puffiness",
        "sunscreens for face with SPF 50",
        "foundations for dry skin",
        "lipsticks that last all day",
        "moisturizers with hyaluronic acid",
        
        # Electronics specific
        "mechanical keyboards for gaming",
        "external hard drives 1TB or more",
        "smartphones with good camera quality",
        "tablets for drawing and art",
        "wireless earbuds with long battery life"
    ]

    quantity_queries = [
        "show me 5 best face creams",
        "recommend 3 laptops for programming",
        "find 10 skincare products under 20€",
        "give me 7 smartphone recommendations",
        "list 4 high-end cameras"
    ]

    mixed_queries = [
        "products for both skin care and electronics",  # Should handle ambiguity
        "best rated items regardless of category",
        "cheap products under 15 euros in any category",
        "most expensive items in your database"
    ]

    natural_queries = [
        "I'm looking for something to help with my dry skin, preferably under 25 bucks",
        "My laptop is slow, need a good processor that won't break the bank",
        "What do you recommend for dark circles? I don't want to spend more than 30 euros",
        "Need a new camera for travel, budget around 500-800 euros",
        "Help me find hair products that actually work for curly hair"
    ]

    boundary_queries = [
        "products exactly at 50 euros",  # Exact price
        "items with 5 star rating",      # Max rating
        "free products",                 # Zero price
        "most expensive item you have",  # No upper limit
        "cheapest electronics"           # No lower limit
    ]

    all_queries = price_queries + domain_queries + complex_queries + specific_queries + quantity_queries + mixed_queries + natural_queries + boundary_queries
    for q in all_queries:
        with open('output.txt', 'w', encoding='utf-8') as f:
            print(f"\nQuery: {q}", file=f)
            print(f"\nQuery: {q}")
            result = rag.retrieve(q)
            df = result['dataframe']  # This assumes your retrieve() returns a dict
            item_count = result['requested_items']  # If you want to log this too
            print(f"Results ({item_count} items): \n", file=f)
            print(df.to_string(), file=f)
            print(df)
            print("\n" + "="*50 + "\n", file=f)  # Separator for readability
