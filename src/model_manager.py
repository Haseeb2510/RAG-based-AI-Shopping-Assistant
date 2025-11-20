import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from functools import lru_cache

class ModelManager:
    _instance = None
    _model = None
    _tokenizer = None
    _last_used = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model_and_tokenizer(cls):
        if cls._model is None or cls._tokenizer is None:
            cls._load_model()
        cls._last_used = time.time()
        return cls._model, cls._tokenizer
    
    @classmethod
    def _load_model(cls):
        print("🔄 Loading model and tokenizer...")
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        
        # Load tokenizer
        cls._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Load model with optimization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        cls._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Add this for faster loading
        )
        
        print("✅ Model and tokenizer loaded!")