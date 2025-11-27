"""
Test script to verify Spanish corpus loading works correctly.
"""

from datasets import load_dataset
from tqdm import tqdm

def test_corpus_loading(corpus_name="mc4", split="validation", max_samples=10):
    """Test loading a Spanish corpus."""
    print(f"\nTesting corpus: {corpus_name} ({split} split)")
    print("="*60)
    
    try:
        if corpus_name == "mc4":
            print("Loading mC4 Spanish subset...")
            dataset = load_dataset(
                "allenai/c4",
                "es",  # Spanish language code
                split=split,
                trust_remote_code=True
            )
            text_field = "text"
            
        elif corpus_name == "wikipedia":
            print("Loading Spanish Wikipedia...")
            dataset = load_dataset(
                "wikipedia",
                "20220301.es",  # Spanish Wikipedia dump
                split=split,
                trust_remote_code=True
            )
            text_field = "text"
            
        elif corpus_name == "oscar":
            print("Loading OSCAR Spanish subset...")
            dataset = load_dataset(
                "oscar",
                "unshuffled_deduplicated_es",  # Spanish subset
                split=split,
                trust_remote_code=True
            )
            text_field = "text"
        else:
            raise ValueError(f"Unknown corpus: {corpus_name}")
        
        print(f"✓ Successfully loaded dataset")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Features: {list(dataset.features.keys())}")
        
        # Show sample texts
        print(f"\nSample texts (first {max_samples}):")
        print("-"*60)
        texts = []
        for i in range(min(max_samples, len(dataset))):
            example = dataset[i]
            text = example.get(text_field, "")
            if text:
                texts.append(text.strip())
                print(f"\n[{i+1}] {text[:200]}..." if len(text) > 200 else f"\n[{i+1}] {text}")
        
        print(f"\n✓ Successfully extracted {len(texts)} texts")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading corpus: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing Spanish Corpus Loading")
    print("="*60)
    
    # Test each corpus
    corpora = ["mc4", "wikipedia", "oscar"]
    
    for corpus in corpora:
        success = test_corpus_loading(corpus_name=corpus, split="validation", max_samples=3)
        if success:
            print(f"\n✓ {corpus} corpus loading test PASSED\n")
        else:
            print(f"\n✗ {corpus} corpus loading test FAILED\n")
    
    print("\n" + "="*60)
    print("Recommendation: Use 'mc4' corpus (most reliable and commonly used)")
    print("="*60)

