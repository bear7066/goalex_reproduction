
import sys
import os
import json

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from cluster_problem import ClusterProblem as Problem
from recursive_cluster import depth2_clustering
from utils import OuterMedusaLLM

def test_baseline():
    print("Setting up baseline test...")
    
    # 1. Create Mock Data
    texts = [
        "The cat sat on the mat.",
        "Dogs are great pets.",
        "I love coding in Python.",
        "Java is verbose but powerful.",
        "Felines are independent animals.",
        "Canines are loyal companions.",
        "Debugging can be fun.",
        "Compilers optimize code."
    ]
    
    goal = "Cluster these sentences into topics about Animals and Programming."
    
    problem = Problem(
        texts=texts,
        goal=goal,
        example_descriptions=["sentences about animals", "sentences about programming"]
    )
    
    print(f"Problem created with {len(texts)} texts.")
    
    # 2. Check if we can run the pipeline
    # We allow the user to see if it runs, but we might catch errors if env vars are missing
    try:
        # Using the refactored logic via recursive_cluster
        # Note: We need to ensure parameters are small to avoid high cost/time
        print("Running depth2_clustering...")
        
        taxonomy = depth2_clustering(
            problem=problem,
            problem_id=999,
            proposer_model="gpt-oss-20b", # Will use OuterMedusaLLM
            assigner_name="gpt-oss-20b",  # Will use OuterMedusaLLM
            num_clusters=2,
            overlap_penalty=0.1,
            not_cover_penalty=0.5,
            random_seed=42,
            iterative_max_rounds=1, # Minimal rounds
            minimal_size=2 # Minimal size to allow clustering
        )
        
        print("\nClustering completed successfully!")
        result_dict = taxonomy.to_dict()
        print("Resulting Taxonomy:")
        print(json.dumps(result_dict, indent=2))
        
        with open("baseline_results.json", "w") as f:
            json.dump(result_dict, f, indent=2)
        print("\nResults saved to baseline_results.json")
        
    except Exception as e:
        print(f"\nExecution failed: {e}")
        print("Note: Ensure OUTER_MEDUSA_ENDPOINT and OUTER_MEDUSA_API_KEY are set.")

if __name__ == "__main__":
    test_baseline()
