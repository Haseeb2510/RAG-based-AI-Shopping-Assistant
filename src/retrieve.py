import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import faiss, joblib, re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.tokenization_embeddings import prepare_data, create_chunks
from src.testing_querys import test_querys
import spacy
from src.paths import TOKENIZATION_DATA, CHUNKS_PATH, WORKED_FOLDER

DOMAIN_KEYWORDS = {
    'Beauty': [
        # Core Categories
        'beauty', 'cosmetics', 'makeup', 'skincare', 'personal', 'grooming', 'toiletries', 'self', 'aesthetics',
        
        # Skincare Products
        'cleanser', 'moisturizer', 'serum', 'toner', 'essence', 'ampoule', 'emulsion', 'facial', 'mist', 'balm',
        'sunscreen', 'sunblock', 'spf', 'bb', 'cc', 'tinted', 'sleeping', 'sheet', 'clay', 'peel', 'wash',
        'exfoliator', 'scrub', 'chemical', 'exfoliant', 'retinol', 'retinoid',
        
        # Skincare Ingredients & Actives
        'hyaluronic', 'acid', 'vitamin', 'niacinamide', 'salicylic', 'glycolic', 'lactic', 'azelaic',
        'ceramide', 'peptide', 'collagen', 'elastin', 'snail', 'mucin', 'propolis', 'centella', 'asiatica', 'green',
        'tree', 'witch', 'hazel', 'aloe', 'vera', 'shea', 'butter', 'jojoba', 'argan', 'rosehip', 'squalane', 'ceramides',
        'antioxidant', 'aha', 'bha', 'pha', 'broad', 'spectrum',
        
        # Skincare Concerns & Benefits
        'anti-aging', 'anti', 'wrinkle', 'fine', 'lines', 'wrinkles', 'firmness', 'elasticity', 'hydration', 'moisture',
        'barrier', 'dehydrated', 'dry', 'oily', 'combination', 'sensitive', 'acne', 'breakouts', 'blackheads', 'whiteheads',
        'pores', 'dark', 'spots', 'hyperpigmentation', 'melasma', 'brightening', 'whitening', 'even', 'tone', 'glow',
        'radiance', 'dullness', 'redness', 'rosacea', 'eczema', 'psoriasis', 'circles', 'puffiness',
        
        # Makeup - Face
        'foundation', 'concealer', 'powder', 'setting', 'pressed', 'loose', 'primer', 'pore', 'filler', 'color', 'corrector',
        'highlighter', 'blush', 'bronzer', 'contour', 'fixer',
        
        # Makeup - Eyes
        'eyeshadow', 'palette', 'eyeliner', 'pencil', 'liquid', 'gel', 'mascara', 'eyebrow', 'brow', 'pomade', 'false',
        'lashes', 'lash', 'glue', 'kajal',
        
        # Makeup - Lips
        'lipstick', 'lip', 'gloss', 'balm', 'stain', 'liner', 'matte', 'plumper',
        
        # Makeup Tools & Accessories
        'brush', 'set', 'blender', 'sponge', 'puff', 'cosmetic', 'organizer', 'sharpener', 'tweezers',
        'curler', 'remover', 'micellar', 'cleansing',
        
        # Hair Care
        'shampoo', 'conditioner', 'hair', 'deep', 'leave', 'protectant', 'mousse', 'wax',
        'texturizing', 'scalp', 'treatment', 'tonic', 'growth', 'loss', 'thinning', 'dandruff', 'split',
        'ends', 'frizz', 'smoothing', 'volumizing', 'curl', 'defining', 'protection', 'purple',
        
        # Hair Styling & Tools
        'dryer', 'blow', 'flat', 'iron', 'straightener', 'curling', 'wand', 'comb', 'hot', 'rollers', 'accessories', 'clip',
        'scrunchies', 'extensions', 'wig', 'topper',
        
        # Body Care
        'body', 'soap', 'shower', 'lotion', 'hand', 'cream', 'foot', 'deodorant', 'antiperspirant',
        'shaving', 'aftershave',
        
        # Fragrance
        'perfume', 'fragrance', 'cologne', 'eau', 'de', 'toilette', 'parfum', 'scent', 'notes', 'family', 'floral', 'woody',
        'fresh', 'oriental',
        
        # Nails
        'nail', 'polish', 'lacquer', 'base', 'coat', 'top', 'quick', 'art', 'stickers', 'file', 'cuticle',
        'manicure', 'pedicure', 'press',
        
        # Professional & Salon
        'salon', 'spa', 'esthetician', 'cosmetologist', 'therapist', 'microdermabrasion', 'dermaplaning',
        'massage', 'eyelash', 'lift', 'lamination', 'microblading', 'permanent', 'artist', 'therapy mask',
        
        # Men's Grooming
        'men', 'beard', 'trimmer', 'kit', 'razor', 'safety', 'electric', 'shaver',
        
        # Natural & Organic
        'clean', 'natural', 'organic', 'cruelty', 'free', 'vegan', 'sustainable', 'green', 'eco', 'friendly', 'packaging',
        'zero', 'waste', 'refillable'
    ],
    'Electronics': [
        # Core Categories
        'electronics', 'technology', 'gadgets', 'devices', 'digital', 'wireless', 'connected',
        
        # Mobile Devices
        'smartphone', 'mobile', 'phone', 'iphone', 'android', 'google', 'pixel', 'samsung', 'galaxy', 'oneplus',
        'tablet', 'ipad', 'e-reader', 'kindle', 'kobo',
        
        # Computers & Laptops
        'laptop', 'notebook', 'ultrabook', 'gaming', 'macbook', 'surface', 'pro', 'chromebook', '2-in-1',
        'desktop', 'computer', 'all-in-one', 'pc', 'workstation', 'server', 'mini', 'stick',
        
        # Computer Components
        'processor', 'cpu', 'gpu', 'graphics', 'card', 'motherboard', 'ram', 'memory', 'ssd', 'hard', 'drive', 'hdd',
        'power', 'supply', 'psu', 'cooling', 'system', 'cooler', 'liquid', 'case', 'chassis', 'thermal', 'paste',
        
        # Computer Peripherals
        'monitor', 'screen', 'ultrawide', '4k', 'curved', 'keyboard', 'mechanical', 'mouse', 'webcam',
        'printer', 'scanner', 'laser', 'inkjet',
        
        # Audio Equipment
        'headphones', 'noise', 'cancelling', 'earbuds', 'true', 'earphones', 'headset', 'speakers', 'bluetooth',
        'speaker', 'soundbar', 'home', 'theater', 'surround', 'sound', 'microphone', 'condenser', 'mic', 'usb',
        
        # Display & TV
        'television', 'tv', 'oled', 'qled', '8k', 'hdr', 'dolby', 'vision', 'projector', 'portable', 'video',
        'wall', 'signage',
        
        # Wearable Technology
        'smartwatch', 'apple', 'watch', 'fitbit', 'fitness', 'tracker', 'band', 'wearable', 'device', 'health',
        'sleep', 'activity', 'glasses', 'ar', 'vr', 'headset', 'virtual', 'reality', 'mixed', 'ring',
        
        # Gaming
        'console', 'playstation', 'xbox', 'nintendo', 'switch', 'chair', 'desk', 'video', 'games', 'game',
        'controller', 'joystick', 'racing', 'wheel', 'flight', 'stick', 'cloud', 'accessories',
        
        # Networking & Connectivity
        'router', 'wifi', 'mesh', 'modem', 'network', 'switch', 'access', 'point', 'range', 'extender', 'ethernet',
        'cable', 'adapter', 'hub', 'dongle', 'thunderbolt', 'dock', 'nas', 'attached', 'storage', 'vpn',
        
        # Storage Solutions
        'external', 'portable', 'flash', 'memory', 'sd', 'microsd', 'cloud', 'data', 'recovery', 'raid', 'array',
        'enclosure', 'optical', 'blu-ray',
        
        # Cameras & Photography
        'camera', 'dslr', 'mirrorless', 'point', 'shoot', 'action', 'gopro', 'drone', 'lens', 'telephoto', 'wide', 'angle',
        'prime', 'zoom', 'tripod', 'bag', 'strap', 'reader', 'photo', 'selfies',
        
        # Smart Home & IoT
        'iot', 'voice', 'assistant', 'alexa', 'siri', 'lighting', 'bulbs', 'plug', 'thermostat', 'lock',
        'security', 'doorbell', 'baby', 'pet', 'vacuum', 'robot', 'fridge', 'oven', 'washer', 'dryer', 'scale',
        'weather', 'station',
        
        # Office Electronics
        'calculator', 'label', 'maker', 'laminator', 'paper', 'shredder', 'postage', 'meter', 'time', 'clock',
        'barcode', 'point', 'sale', 'pos', 'cash', 'register', 'presentation', 'clicker', 'document',
        
        # Automotive Electronics
        'car', 'stereo', 'head', 'unit', 'speakers', 'subwoofer', 'amplifier', 'dash', 'cam', 'gps', 'navigator',
        'charger', 'inverter', 'jump', 'starter', 'tire', 'pressure', 'backup', 'carplay', 'auto',
        
        # Health & Medical Electronics
        'blood', 'pressure', 'glucometer', 'pulse', 'oximeter', 'thermometer', 'body', 'analyzer', 'electric',
        'toothbrush', 'water', 'flosser', 'removal', 'therapy', 'massage', 'gun', 'hearing', 'aid',
        'tens', 'medical', 'alert',
        
        # Components & Parts
        'battery', 'rechargeable', 'bank', 'charging', 'hdmi', 'connector', 'socket', 'relay', 'transistor',
        'capacitor', 'resistor', 'pcb', 'arduino', 'raspberry', 'pi',
        
        # Emerging Technologies
        'ai', 'machine', 'learning', 'computer', 'vision', 'edge', 'computing', 'quantum', 'blockchain',
        'cryptocurrency', 'mining', '3d', 'printer', 'scanner', 'robotics', 'drones', 'autonomous', 'vehicles',
        'biometric', 'facial', 'recognition', 'fingerprint', 'retina',
        
        # Accessories & Carrying
        'case', 'protector', 'sleeve', 'backpack', 'protective', 'cover', 'stand',
        'mount', 'management', 'surge'
    ]
}

class RAGRetriever:
    def __init__(self) -> None:
        self.all_index, self.beauty_index, self.elec_index, self.meta, self.table = self._load_indices()
        self.model_path = self.meta.get("model_name")
        self.model = SentenceTransformer(self.model_path)
        self.full_df = self._load_full_data() 
        self.nlp = spacy.load("en_core_web_sm")
    
    def _load_indices(self) -> tuple[faiss.Index, faiss.Index, faiss.Index, dict, pd.DataFrame]:
        try:
            ia = faiss.read_index(os.path.join(TOKENIZATION_DATA, "faiss_all.index"))
            ib = faiss.read_index(os.path.join(TOKENIZATION_DATA, "faiss_beauty.index"))
            ie = faiss.read_index(os.path.join(TOKENIZATION_DATA, "faiss_electronics.index"))

            m = joblib.load(os.path.join(TOKENIZATION_DATA, "meta.joblib"))
            t = pd.read_parquet(m['doc_table_path'])
            return ia, ib, ie, m, t
        except FileNotFoundError as e:
            print(f"❌ Index files not found: {e}")
            print("💡 Run prepare_data() first to create indices")
            raise

    def _load_full_data(self):
        if os.path.exists(CHUNKS_PATH):
            self.full_df = pd.read_parquet(CHUNKS_PATH)
            full_df = self.full_df
        else:    
            full_df = pd.read_parquet(os.path.join(WORKED_FOLDER, "cleaned_corpus_rich.parquet"), columns=[
                    "main_category", "title", "average_rating", "price", "categories", "domain", "combined_text"
            ])
            full_df["price"] = (
                full_df["price"]
                .astype(str)
                .str.replace("€", "", regex=False)
                .str.replace(",", ".", regex=False)
                .str.extract(r"([\d.]+)")[0]   # extract numeric part
                .astype(float)
            )

            full_df["average_rating"] = (
                full_df["average_rating"]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.extract(r"([\d.]+)")[0]
                .astype(float)
            )

            full_df['doc_id'] = np.arange(len(full_df), dtype=np.int64)
            full_df = create_chunks(full_df)
            self.full_df = full_df
        return full_df

    def parse_rating(self, q: str) -> tuple[str, float] | None:
        """
        Enhanced rating parser that handles multiple patterns
        Returns: (operator, value) or None
        """
        
        # Pattern 1: Explicit rating patterns
        patterns = [
            # "above 4.2 rating", "over 4 stars", "greater than 4.5 stars"
            r"(above|over|greater than|more than|at least|minimum|min)\s*([0-9.]+)\s*(?:stars?|\+|\*|rating|rated)?",
            
            # "4+ stars", "4.5+ rating", "5 star rating"
            r"([0-9.]+)\s*[\+*]\s*(?:stars?|rating)",
            
            # "highly rated", "good ratings", "excellent reviews"
            r"(highly rated|excellent|awesome|great|good)\s+(?:ratings?|reviews?|stars?)",
            
            # "minimum 4 stars", "at least 4.5 rating"
            r"(minimum|min|at least)\s*([0-9.]+)\s*(?:stars?|rating)",
            
            # "rated 4.5 or higher", "4 stars and above"
            r"(?:rated|rating|stars?)\s*([0-9.]+)\s*(?:or\s+(?:higher|more|above|better)|and\s+(?:above|up))",
            
            # "5 star products", "4.5 rated items"
            r"([0-9.]+)\s*(?:star|rated)\s+(?:products?|items?|options?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, q, re.IGNORECASE)
            if match:
                if match.group(0) in ["highly rated", "excellent", "awesome", "great", "good"]:
                    return (">=", 4.5)  # Map quality words to numeric ratings
                elif len(match.groups()) >= 2 and match.group(2):
                    try:
                        rating_value = float(match.group(2))
                        if 1 <= rating_value <= 5:  # Validate rating range
                            return (">=", rating_value)
                    except ValueError:
                        continue
        
        # Pattern 2: Look for standalone high numbers that could be ratings
        standalone_ratings = re.findall(r'\b([4-5](?:\.\d+)?)\s*(?:stars?|rating|rated)?\b', q)
        for rating_str in standalone_ratings:
            try:
                rating_value = float(rating_str)
                if rating_value >= 4.0:  # Only consider high numbers as potential ratings
                    return (">=", rating_value)
            except ValueError:
                continue
        
        return None

    def extract_price_filters(self, query: str):
        """
        Extract price constraints from natural language.
        Returns one of:
            (min_price, max_price)
            (None, max_price)   -> for "under X"
            (min_price, None)   -> for "over X"
            (None, None)        -> if no price found
        """
        
        q = query.lower().replace(",", "")  # remove thousand separators like 1,000

        # 1) RANGE PATTERNS: "50-150", "between 50 and 150", etc.
        range_patterns = [
            r'(\d+)\s*-\s*(\d+)',
            r'between\s+(\d+)\s+and\s+(\d+)',
            r'(\d+)\s+to\s+(\d+)',
            r'from\s+(\d+)\s+to\s+(\d+)',
        ]

        for pattern in range_patterns:
            match = re.search(pattern, q)
            if match:
                low = float(match.group(1))
                high = float(match.group(2))
                return "range", [low, high]

        # 2) UNDER / BELOW / CHEAP / BUDGET
        # Handles: "under 200", "for under $200", "under € 200", "under-200"
        under_pattern = r"(under|below|less than|cheap|affordable|budget)[^\d]*(\$|€)?\s*(\d+)"
        match = re.search(under_pattern, q)
        if match:
            max_price = float(match.group(3))
            return "<", max_price    # no min_price

        # 3) OVER / ABOVE / MORE THAN / EXPENSIVE
        # Handles: "over 200", "above $300", "more than €150"
        over_pattern = r"(over|above|more than|expensive|premium|high-end)[^\d]*(\$|€)?\s*(\d+)"
        match = re.search(over_pattern, q)
        if match:
            min_price = float(match.group(3))
            return ">", min_price

        # 4) EXACT PRICE MENTION
        # Handles: "for 200 euro", "at 150€", "200 usd"
        # If there's exactly ONE price → treat as exact price filter
        exact_pattern = r'(?:for|cost|price|at)\s*(\$|€)?\s*(\d+)(?:\s*(euro|€|eur|usd|dollar))?'
        matches = re.findall(exact_pattern, q)

        # Remove duplicates like ("200","€") appearing twice
        numbers = {float(m[1]) for m in matches if m[1].isdigit()}

        if len(numbers) == 1:
            price = numbers.pop()
            return "=", price

        # No price filters
        return None, None


    def get_domain_hint(self, query: str) -> dict:
        result =  {
            "domain_hint": None
        }
        q = query.lower()
        
        clauses = re.split(r'[,;]|\band\b|\bor\b', query)
        clauses = [clause.strip() for clause in clauses if clause.strip()]

        beauty_score = sum(1 for word in DOMAIN_KEYWORDS["Beauty"] if re.search(rf"\b{re.escape(word)}\b", q))
        electronics_score = sum(1 for word in DOMAIN_KEYWORDS["Electronics"] if re.search(rf"\b{re.escape(word)}\b", q))
        if len(q) < 100 and len(clauses) < 2:
            if beauty_score > electronics_score and beauty_score > 0:
                result["domain_hint"] = "Beauty"                    # type: ignore
            elif electronics_score > beauty_score and electronics_score > 0:
                result["domain_hint"] = "Electronics"               # type: ignore
        else:
            # Set minimum score requirements
            min_score_threshold = 2
            total_score = beauty_score + electronics_score
            if total_score >= min_score_threshold:
                ratio = beauty_score / electronics_score if electronics_score > 0 else float('inf')
                if ratio >= 3:  # Beauty is 3x more dominant
                    result["domain_hint"] = ["Beauty"]                  # type: ignore
                elif ratio <= 0.33:  # Electronics is 3x more dominant
                    result["domain_hint"] = ["Electronics"]             # type: ignore
                elif ratio >= 0.5 and ratio <= 2:  # Relatively balanced
                    result["domain_hint"] = ["Beauty", "Electronics"]   # type: ignore
                else:
                    result["domain_hint"] = [None]                      # type: ignore
            else:
                result["domain_hint"] = [None]  # Not enough domain-specific words  # type: ignore
        return result

    def parse_query(self, query: str) -> dict:
        """
        Parse natural-language query to extract:
        - numeric filters (price, rating)
        - domain hints (beauty, electronics)
        Returns a dict usable by retrieve_advanced().
        """
        q = query.lower()
        result = {
            "price_filter": None,
            "rating_filter": None,
            "num_products": None,
        }

        # --- Detect product patterns ---
        patterns = [
            # Pattern 1: "recommend 20", "suggest 10 products", "find 5 items" 
            r'(?:recommend|suggest|show|find|give|list|recomend)\s+(?:me\s+)?(\d+)\s*(?:products?|items?|options?|recommendations?)?\b',
            
            # Pattern 2: "I want 10 products", "need 5 items" 
            r'(?:want|need|looking for|searching for)\s+(?:.*?)?(\d+)\s*(?:products?|items?|options?)?\b',
            
            # Pattern 3: "10 products for..." 
            r'(\d+)\s*(?:products?|items?)\s+(?:for|about|regarding)',
            
            # Pattern 4: Number at the end "show me 20"
            r'(?:products?|items?|recomend|recommend)\s+(\d+)$',
            
            # Pattern 5: Simple number after verb "get 10", "find 15"
            r'\b(?:get|find|recomend|recommend)\s+(\d+)\b',
            
            # Pattern 6: Any standalone number that might be quantity
            r'\b(\d+)\s*(?:products?|items?|options?)\b',
        ]
        # Remove price ranges and standalone price numbers to avoid false matches
        cleaned_q = re.sub(r'\d+\s*-\s*\d+', '', q)          # remove "500-800"
        cleaned_q = re.sub(r'\d+\s*(?:euro|€|dollar|usd)', '', cleaned_q)  # remove "500 euro"
        cleaned_q = re.sub(r'(?:over|under|below|less than|more than)\s*\d+', '', cleaned_q)

        for pattern in patterns:
            match = re.search(pattern, cleaned_q)
            if match:
                result["num_products"] = int(match.group(1))        # type: ignore
                break

        # If no pattern matched, try to find any number in the query
        if result["num_products"] is None:
            # Look for any number that might indicate quantity
            number_match = re.search(r'\b(\d+)\s+(?:products?|items?|options?)\b', q)
            if number_match:
                result["num_products"] = int(number_match.group(1)) # type: ignore

        # --- Detect price patterns ---
        # Example: "under 30", "below 50 euro", "less than 20€"
        result["price_filter"] = (self.extract_price_filters(q))   # type: ignore


        # --- Detect rating pattern ---
        # Example: "above 4.2 rating", "over 4 stars"
        result["rating_filter"] = self.parse_rating(q)  # type: ignore

        return result

    def query_lemmas(self, query: str) -> set[str]:
        doc = self.nlp(query.lower())
        return {token.lemma_ for token in doc if token.is_alpha}

    def query_decomposition(self, query: str, domains: tuple|list) -> dict:
        """
        Improved query decomposition using sentence parsing and intent detection
        """
        decomposed_queries = {}
        
        # Split query into clauses
        clauses = re.split(r'[,;]|\band\b|\bor\b', query)
        clauses = [clause.strip() for clause in clauses if clause.strip()]

        # Initialize with fallbacks
        beauty_clause = query
        electronics_clause = query
        assigned_clauses = set()

        # First pass: find the best clause for each domain
        beauty_candidates = []
        electronics_candidates = []
        
        for clause in clauses:
            clause_lemmas = self.query_lemmas(clause)

            beauty_terms = sum(1 for kw in DOMAIN_KEYWORDS["Beauty"] if kw in clause_lemmas)
            electronics_terms = sum(1 for kw in DOMAIN_KEYWORDS["Electronics"] if kw in clause_lemmas)
            
            # Store candidates with their scores
            beauty_candidates.append((clause, beauty_terms - electronics_terms))
            electronics_candidates.append((clause, electronics_terms - beauty_terms))
        
        # Sort candidates by domain relevance (highest score first)
        beauty_candidates.sort(key=lambda x: x[1], reverse=True)
        electronics_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Assign best available clauses to each domain
        for domain in domains:
            if domain == 'Beauty':
                # Find the best beauty clause that hasn't been assigned
                for clause, score in beauty_candidates:
                    if clause not in assigned_clauses and score >= 0:
                        beauty_clause = clause
                        assigned_clauses.add(clause)
                        break
            else:  # Electronics
                # Find the best electronics clause that hasn't been assigned
                for clause, score in electronics_candidates:
                    if clause not in assigned_clauses and score >= 0:
                        electronics_clause = clause
                        assigned_clauses.add(clause)
                        break
        
        # Assign to domains
        for domain in domains:
            if domain == 'Beauty':
                subquery = beauty_clause
                confidence = 0.8 if beauty_clause != query else 0.1
            else:
                subquery = electronics_clause
                confidence = 0.8 if electronics_clause != query else 0.1
            
            decomposed_queries[domain] = {
                'subquery': subquery,
                'domain_terms': [],
                'confidence': confidence
            }
        
        return decomposed_queries

    def enhance_product_filtering(self, query: str, results_df: pd.DataFrame):
        query_lower = query.lower()
        
        # Strict laptop filtering
        if any(word in query_lower for word in ['laptop', 'notebook', 'macbook']):
            results_df = results_df[
                (results_df['title'].str.contains('laptop|notebook|macbook', case=False, na=False)) &
                (~results_df['title'].str.contains('case|bag|sleeve|cover|accessory', case=False, na=False)) &
                (results_df['categories'].str.contains('laptop', case=False, na=False))
            ]
        
        # Strict camera filtering  
        if any(word in query_lower for word in ['camera', 'dslr', 'mirrorless']):
            results_df = results_df[
                (results_df['title'].str.contains('camera|dslr|mirrorless', case=False, na=False)) &
                (~results_df['title'].str.contains('security|surveillance|system|dvr|nvr', case=False, na=False))
            ]
        
        return results_df
    
    def extract_keywords(self, query: str) -> set:
        """Extract meaningful keywords from the query."""
        # Remove common stop words and split into keywords
        stop_words = {'find', 'me', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = {word for word in words if word not in stop_words and len(word) > 2}
        return keywords

    def remove_redundant_text(self, text: str, row: pd.Series, columns_to_compare: list) -> str:
        """Remove text that's already present in other specified columns."""
        cleaned_text = text
        
        for column in columns_to_compare:
            if column in row and pd.notna(row[column]): # type: ignore
                column_value = str(row[column])
                # Remove exact matches and similar phrases
                column_words = re.findall(r'\b\w+\b', column_value.lower())
                for word in column_words:
                    if len(word) > 3:  # Only consider words longer than 3 characters
                        pattern = r'\b' + re.escape(word) + r'\b'
                        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra spaces and punctuation
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = re.sub(r'[.,;]+(\s*[.,;])*', '. ', cleaned_text)
        return cleaned_text.strip()

    def extract_query_relevant_content(self, text: str, query_keywords: set) -> str:
        """Extract and emphasize parts of text relevant to the query."""
        if not query_keywords:
            return text
        
        sentences = re.split(r'[.!?]+', text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if sentence contains any query keywords
            if any(keyword in sentence_lower for keyword in query_keywords):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:3])  # Return top 3 relevant sentences
        else:
            # If no direct matches, return the beginning of the text
            return text[:150] + "..." if len(text) > 150 else text
        
    def clean_text_column(self, df: pd.DataFrame, query: str, columns_to_compare: list = ['title', 'categories', 'domain']) -> pd.DataFrame:
        """
        Clean the text column by removing redundant information already present in other columns
        and focus on content relevant to the query.
        """
        if df.empty:
            return df
        
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Extract query keywords
        query_keywords = self.extract_keywords(query)
        
        cleaned_texts = []
        for idx, row in df_clean.iterrows():
            original_text = str(row['text'] if pd.notna(row['text']) else '')
            
            # Remove text that's already in other columns
            cleaned_text = self.remove_redundant_text(original_text, row, columns_to_compare)
            
            # Focus on query-relevant content
            cleaned_text = self.extract_query_relevant_content(cleaned_text, query_keywords)
            
            # If we removed too much, keep some original context
            if len(cleaned_text.strip()) < 20:
                cleaned_text = original_text[:200] + "..." if len(original_text) > 200 else original_text
            
            cleaned_texts.append(cleaned_text.strip())
        
        df_clean['text'] = cleaned_texts
        return df_clean

    def retrieve_debug(self, merged:pd.DataFrame, final_results: pd.DataFrame, filtered_results:pd.DataFrame, initial_retrieval_count: int, top_k_each: int, query:str):
        print("="*50)
        print("Final_results lenght: ", len(final_results))
        print("Initial_retrieval_lenght lenght: ", initial_retrieval_count)
        print("="*50)
        print("="*50)
        print("Query: ", query)
        print("="*50)
        print("\n---Before filtering---")
        print("\nBefore filtering lenght: ", len(merged))
        print(merged.head(top_k_each))
        print("\n---After filtering---")
        print("\nAfter filtering lenght: ", len(filtered_results))
        print("\n--- Duplicate Analysis ---")
        print(f"Total rows: {len(filtered_results)}")
        print(f"Duplicates (all columns): {sum(filtered_results.duplicated())}")
        print(f"Duplicate doc_ids: {sum(filtered_results.duplicated(subset=['doc_id']))}")
        print(f"Duplicate (doc_id + chunk_id): {sum(filtered_results.duplicated(subset=['doc_id', 'chunk_id']))}")
        print(f"Unique doc_ids: {filtered_results['doc_id'].nunique()}")
        print(filtered_results.head(top_k_each))
        print("\n---FInal results---")
        print("\nFinal lenght: ", len(final_results))
        print(final_results.head(top_k_each))

    def score_boost(self, row, query_lower):
        boost = 0.0

        if row["title"] and any(w in row["title"].lower() for w in query_lower.split()):
            boost += 0.03

        if row["categories"] and any(w in row["categories"].lower() for w in query_lower.split()):
            boost += 0.02

        return boost


    def retrieve_advanced(
        self,
        query: str,
        top_k_each=3,
        fake_top_k_each: int|None=None,
        price_filter=None,      # e.g., ("<", 30)
        rating_filter=None,     # e.g., (">", 4.0)
        domain: str|None=None,
        show_metadata=True
    ) -> pd.DataFrame:
        """
        Enhanced retrieval with:
        - optional numeric filtering (price, rating)
        - balanced Beauty/Electronics retrieval
        - inclusion of product metadata
        """

        
        # Get data
        df = self.full_df

        # --- Modify query based on price filter ---
        # Add domain-specific context
        if domain == "Beauty":
            augmented_query = f"{query} skincare makeup beauty cosmetic"
        elif domain == "Electronics":
            augmented_query = f"{query} electronics tech gadget device"
        else:    
            augmented_query = query

        if price_filter:
            op, val = price_filter
            if op == "<":
                augmented_query = augmented_query + " cheap affordable budget"
            elif op == ">":
                augmented_query = augmented_query + " premium high-end expensive"
            elif op == "range":
                augmented_query = augmented_query + " mid-range"

        # --- Load all components ---
        q = self.model.encode([augmented_query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")

        # --- Helper: apply numeric filter on metadata ---
        def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
            filtered_df = df.copy()
            if price_filter:
                op, val = price_filter
                if op == "<":
                    filtered_df = filtered_df[filtered_df["price"] < val]
                elif op == ">":
                    filtered_df = filtered_df[filtered_df["price"] > val]
                elif op == "range":
                    low, high = val
                    if low is not None and high is not None:
                        filtered_df = filtered_df[(filtered_df["price"] >= low) & (filtered_df["price"] <= high)]
                    elif low is not None:
                        filtered_df = filtered_df[filtered_df["price"] >= low]
                    elif high is not None:
                        filtered_df = filtered_df[filtered_df["price"] <= high]
                elif op == "=":
                    filtered_df = filtered_df[filtered_df["price"] == val]

            if rating_filter:
                op, val = rating_filter
                if op == ">":
                    filtered_df = filtered_df[filtered_df["average_rating"] > val]
                elif op == ">=":
                    filtered_df = filtered_df[filtered_df["average_rating"] >= val]
                filtered_df = filtered_df.sort_values("average_rating", ascending=False)
            return filtered_df
        
        
        # Retrieve more items to have better candidates for filtering
        # Estimate how much filtering you'll do
        if fake_top_k_each is not None:
            initial_retrieval_count = max(fake_top_k_each * 50, 200)
        else:
            initial_retrieval_count = max(top_k_each * 50, 200)

        # --- Perform retrieval ---
        def _domain_search(q, retrieval_count: int, domain: str|None=None) -> pd.DataFrame:
            if domain:
                if domain == "Beauty":
                    scores, ids = self.beauty_index.search(q, retrieval_count)  # type: ignore
                    rows = self.table.iloc[self.meta["beauty_indices"][ids[0]]].copy()
                else:  # domain == "Electronics"
                    scores, ids = self.elec_index.search(q, retrieval_count)  # type: ignore
                    rows = self.table.iloc[self.meta["electronics_indices"][ids[0]]].copy()
                
                rows["score"] = scores[0]
                combined_rows= rows
                
            else:
                scores, ids = self.all_index.search(q, retrieval_count)# type: ignore
                combined_rows = self.table.iloc[ids[0]].copy()
                combined_rows["score"] = scores[0]

            return combined_rows
        
        combined_rows = _domain_search(q, initial_retrieval_count, domain)

        # --- Join back with full metadata for context richness ---
        merged = combined_rows.merge(
            df,
            on="doc_id",
            how="left",
            suffixes=("", "_full")
        )

        # Boost score
        merged["score"] = merged.apply(lambda r: r["score"] + self.score_boost(r, query.lower()), axis=1)

        # --- Apply filters ---
        filtered_results = self.enhance_product_filtering(query, merged)
        filtered_results = apply_filters(filtered_results)

        # --- Handle empty results ---
        if filtered_results.empty:
            # Return filtered results
            final_results = merged.drop_duplicates(subset=["doc_id"]).reset_index(drop=True).sort_values(["score"]).head(top_k_each).copy()
            final_results['filter_warning'] = True
        else:
            # Return filtered results
            final_results = filtered_results.drop_duplicates(subset=["doc_id"]).reset_index(drop=True).head(top_k_each).copy()
            final_results['filter_warning'] = False
        
        # --- Format output ---
        cols = ["doc_id", "domain", "chunk_id", "score", "title", "price", "average_rating", "categories", "text"]
        if not show_metadata:
            cols = ["doc_id", "title", "price", "average_rating", "categories", "text"]

        final_results.fillna({"categories": "General", "average_rating": 0}, inplace=True)


        # If results are lower then requested by the query or not filtered then get more products with increased search 
        if (len(final_results) < top_k_each) or (final_results['filter_warning'].any() == True):
            print(f"⚠️ Only {len(final_results)} results found with {initial_retrieval_count} products count, need {top_k_each}. Expanding search...")
            
            attempt = 0
            max_attempts = 3
            
            while attempt < max_attempts:
                attempt += 1
                return self.retrieve_advanced(
                    query=query,
                    top_k_each=top_k_each,
                    fake_top_k_each=initial_retrieval_count*2,
                    price_filter=price_filter,
                    rating_filter=rating_filter,
                    domain=domain,
                    show_metadata=True
                )

        if final_results['filter_warning'].any():
                print("⚠️ Some results may not match price/rating filters")
        
        if show_metadata:
            final_results = (
                filtered_results[cols]
                    .sort_values(["score", "average_rating"], ascending=[False, False])
                    .drop_duplicates("doc_id")
                    .head(top_k_each)
                    .reset_index(drop=True)
            )
        else:
            final_results = (
                filtered_results[cols]
                    .sort_values(["average_rating"], ascending=[False])
                    .drop_duplicates("doc_id")
                    .head(top_k_each)
                    .reset_index(drop=True)
            )
            final_results = final_results.drop(columns=["doc_id"])
            final_results = self.clean_text_column(filtered_results, query)
        # self.retrieve_debug(merged, final_results, filtered_results, initial_retrieval_count, top_k_each, query)

        return final_results.head(top_k_each).reset_index(drop=True)

    def retrieve_with_decomposed_queries(
        self,
        query: str,
        top_k_each: int|None=None,
        domain_hint: list | tuple | None = None,
        show_metadata: bool = True
    ) -> dict:
        """
        Enhanced retrieval using query decomposition.
        Handles multi-domain queries by splitting into domain-specific subqueries.
        """

        # MULTI-DOMAIN CASE ---------------------------------------------
        if isinstance(domain_hint, (list, tuple)) and len(domain_hint) > 1:

            decomposed = self.query_decomposition(query, domain_hint)

            results = []
            top_k = 0
            parseds = {}
            for domain in domain_hint:
                info = decomposed.get(domain, None)
                if info is None:
                    continue

                # Decide which subquery to use
                if info["confidence"] >= 0.5:  # stronger threshold
                    search_query = info["subquery"]
                else:
                    search_query = query  # fallback

                # Parse domain-specific subquery
                parsed = self.parse_query(search_query)
                price_filter = parsed["price_filter"]
                rating_filter = parsed["rating_filter"]

                # per-domain top_k logic
                k_each = top_k_each or parsed["num_products"] or 5
                k_each = min(k_each, 50)

                df = self.retrieve_advanced(
                    query=search_query,
                    top_k_each=k_each,
                    price_filter=price_filter,
                    rating_filter=rating_filter,
                    domain=domain,
                    show_metadata=show_metadata
                )

                if not df.empty:
                    top_k += k_each
                    results.append(df)
                    parseds[domain] = parsed

            # Combine AND deduplicate by doc_id
            if results:
                combined = pd.concat(results, ignore_index=True)
                combined = combined.drop_duplicates("doc_id").reset_index(drop=True)

                return {
                    "dataframe": combined,
                    "requested_items": top_k,
                    "parsed_query": parseds  # Optional: include the full parsed query for debugging
                }

            return {}  # no results found


        # SINGLE DOMAIN CASE --------------------------------------------
        else:
            parsed = self.parse_query(query)
            price_filter = parsed["price_filter"]
            rating_filter = parsed["rating_filter"]

            k_each = top_k_each or parsed["num_products"] or 5
            k_each = min(k_each, 50)

            # if domain_hint exists → use first
            domain = None
            if isinstance(domain_hint, (list, tuple)) and domain_hint:
                domain = domain_hint[0]

            results = self.retrieve_advanced(
                query=query,
                top_k_each=k_each,
                price_filter=price_filter,
                rating_filter=rating_filter,
                domain=domain,
                show_metadata=show_metadata
            )
        
            return {
                "dataframe": results,
                "requested_items": top_k_each,
                "parsed_query": parsed  # Optional: include the full parsed query for debugging
            }

    def retrieve(self, query: str, show_metadata= True) -> dict:
        domain_hint = self.get_domain_hint(query)
        results =  self.retrieve_with_decomposed_queries(
            query=query,
            domain_hint=domain_hint["domain_hint"],
            show_metadata=show_metadata
        )
        return results



if __name__ == '__main__':
    
    # Chek if indices exist, if not create them first
    indices_exist = all(os.path.exists(os.path.join(TOKENIZATION_DATA, f"faiss_{name}.index")) 
                       for name in ["all", "beauty", "electronics"])
    if not indices_exist:
        print("🔨 First-time setup: Creating indices...")
        prepare_data()                

    rag = RAGRetriever()
    
    test_querys(rag)

    print("Search Beauty & Electronics products (e for exit).")
    while True:
        query = input("Search: ")
        if query.lower() == "e":
            break
        else:
            print("\n" + "="*50 + "\n")  # Separator for readability
            print(f"\nQuery: {query}")
            result = rag.retrieve(query, False)
            df = result['dataframe']  # This assumes your retrieve() returns a dict
            item_count = result['requested_items']  # If you want to log this too
            print(f"Results ({len(df)} items): \n")
            print(df)
            print("\n" + "="*50 + "\n")  # Separator for readability