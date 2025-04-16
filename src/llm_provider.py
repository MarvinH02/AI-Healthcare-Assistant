from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def get_tinyllama_llm(temperature=0.4, max_new_tokens=500):
    """
    Initialize and return a TinyLLama model for text generation.
    
    Args:
        temperature (float): Controls randomness in generation (lower = more deterministic)
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        LangChain compatible LLM
    """
    # Model ID from Hugging Face
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading model: {model_id}")
    
    # Check if CUDA is available and set device accordingly
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model with simpler configuration to avoid device_map issues
    try:
        # Attempt to load with minimal settings
        if device == "cuda":
            # For GPU - try to use 8-bit quantization if bitsandbytes is available
            try:
                import bitsandbytes
                print("Using 8-bit quantization")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    load_in_8bit=True
                )
            except ImportError:
                print("bitsandbytes not available, loading model without quantization")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16
                ).to(device)
        else:
            # For CPU - load with defaults
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            
    except Exception as e:
        print(f"Error loading model with optimizations: {e}")
        print("Falling back to basic model loading...")
        # Fallback to simplest loading method
        model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # Create text generation pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Create LangChain wrapper around the pipeline
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    
    return llm