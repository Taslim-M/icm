from icm import ICMSearcher, load_icm_dataset

# Load dataset
dataset = load_icm_dataset("truthful_qa", task_type="truthfulqa")

# Create searcher
searcher = ICMSearcher(
    model_name="distilgpt2",
    alpha=50.0,
    max_iterations=100
)

# Run ICM search
result = searcher.search(dataset, max_examples=50)

# Access results
print(f"Generated {len(result.labeled_examples)} labeled examples")
print(f"Final score: {result.score:.4f}")