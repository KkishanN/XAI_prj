import json
import random
import os

# --- Configuration ---
LP_FILE_PATH = 'Family_LP/lps.json'
OUTPUT_DIR = 'data'
TRAIN_SPLIT_RATIO = 0.6
RANDOM_SEED = 42

def create_files_with_ids():
    """Generates complete, training, and test TSV files with an explicit ID column."""
    print("--- Generating Data Files with Explicit IDs ---")

    # 1. Load source data
    with open(LP_FILE_PATH, 'r') as f:
        data = json.load(f)
    uncle_data = data['problems']['Uncle']
    pos_examples = sorted(list(uncle_data['positive_examples']))
    neg_examples = sorted(list(uncle_data['negative_examples']))
    all_uris = sorted(pos_examples + neg_examples)

    # 2. Create the master ID mapping
    uri_to_id_map = {uri: i for i, uri in enumerate(all_uris)}
    
    # 3. Create the completeDataset.tsv file
    complete_path = os.path.join(OUTPUT_DIR, "completeDataset.tsv")
    with open(complete_path, 'w') as f:
        f.write("id\turi\tlabel\n")
        for uri in all_uris:
            label = 1 if uri in pos_examples else 0
            node_id = uri_to_id_map[uri]
            f.write(f"{node_id}\t{uri}\t{int(label)}\n")
    print(f"Created complete dataset with {len(all_uris)} entries: {complete_path}")

    # 4. Create stratified train/test splits using the master IDs
    random.seed(RANDOM_SEED)
    random.shuffle(pos_examples)
    random.shuffle(neg_examples)

    pos_split_idx = int(len(pos_examples) * TRAIN_SPLIT_RATIO)
    train_pos = pos_examples[:pos_split_idx]
    test_pos = pos_examples[pos_split_idx:]

    neg_split_idx = int(len(neg_examples) * TRAIN_SPLIT_RATIO)
    train_neg = neg_examples[:neg_split_idx]
    test_neg = neg_examples[neg_split_idx:]
    
    # 5. Write trainingSet.tsv
    train_path = os.path.join(OUTPUT_DIR, "trainingSet.tsv")
    with open(train_path, 'w') as f:
        f.write("id\turi\tlabel\n")
        for uri in sorted(train_pos):
            f.write(f"{uri_to_id_map[uri]}\t{uri}\t1\n")
        for uri in sorted(train_neg):
            f.write(f"{uri_to_id_map[uri]}\t{uri}\t0\n")
    print(f"Created training set: {train_path}")

    # 6. Write testSet.tsv
    test_path = os.path.join(OUTPUT_DIR, "testSet.tsv")
    with open(test_path, 'w') as f:
        f.write("id\turi\tlabel\n")
        for uri in sorted(test_pos):
            f.write(f"{uri_to_id_map[uri]}\t{uri}\t1\n")
        for uri in sorted(test_neg):
            f.write(f"{uri_to_id_map[uri]}\t{uri}\t0\n")
    print(f"Created test set: {test_path}")

if __name__ == "__main__":
    create_files_with_ids()
