import json
import os
import time

# Added imports for data handling
import pandas as pd
from sklearn.model_selection import train_test_split

# Ontolearn and OWL-related imports from your original code
from ontolearn.concept_learner import EvoLearner
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import F1, Accuracy
from owlapy.model import IRI, OWLNamedIndividual
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def calculate_metrics(predictions_data):
    """Calculate evaluation metrics for EvoLearner predictions."""
    metrics_dict = {}
    
    for concept_name, examples in predictions_data.items():
        concept_individuals = examples["concept_individuals"]
        positive_examples = set(examples["positive_examples_test"]["uri"].tolist())
        negative_examples = set(examples["negative_examples_test"]["uri"].tolist())

        all_examples = positive_examples.union(negative_examples)

        true_labels = [1 if item in positive_examples else 0 for item in all_examples]
        pred_labels = [1 if item in concept_individuals else 0 for item in all_examples]

        # Calculate accuracy
        accuracy = accuracy_score(true_labels, pred_labels)

        # Calculate precision, recall, f1-score, and support for binary class
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="binary", zero_division=0
        )

        # Store metrics for this concept
        metrics_dict[concept_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1_score,
        }

    return metrics_dict


def train_evo(learning_problems, kg=None):
    """
    Your original function to train EvoLearner on a set of learning problems.
    """
    if learning_problems is None:
        print("No Learning problems provided. stopping training EvoLearner")
        return

    kg_path = f"data/{kg}.owl"
    explanation_dict = {}
    predictions_data = {}

    if not os.path.exists(kg_path):
        print(
            f"Dataset not found at: {kg_path}. Please provide dataset {kg} at the designated location"
        )
        return None
    else:
        t0 = time.time()
        # A valid KG is required here.
        try:
            target_kb = KnowledgeBase(path=kg_path)
        except Exception as e:
            print(f"Error loading Knowledge Base from {kg_path}: {e}")
            print("Please ensure it is a valid OWL file.")
            return None

        for str_target_concept, examples in learning_problems.items():
            positive_examples = set(examples["positive_examples_train"]["uri"].tolist())
            negative_examples = set(examples["negative_examples_train"]["uri"].tolist())
            print("\nTarget concept: ", str_target_concept)

            typed_pos = set(map(OWLNamedIndividual, map(IRI.create, positive_examples)))
            typed_neg = set(map(OWLNamedIndividual, map(IRI.create, negative_examples)))
            lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)

            model = EvoLearner(
                knowledge_base=target_kb, max_runtime=1000, quality_func=F1()
            )
            model.fit(lp, verbose=False)

            # Get Top n hypotheses
            hypotheses = list(model.best_hypotheses(n=3))
            print("Top hypotheses found:")
            [print(f"  - {h}") for h in hypotheses]

            # The rest of your code for evaluation and explanation extraction
            if not hypotheses:
                print("EvoLearner could not find a hypothesis.")
                continue

            positive_examples_test = set(examples["positive_examples_test"]["uri"].tolist())
            negative_examples_test = set(examples["negative_examples_test"]["uri"].tolist())

            best_concept = hypotheses[0].concept
            concept_ind = set(
                [
                    indv.get_iri().as_str()
                    for indv in target_kb.individuals_set(best_concept)
                ]
            )
            concept_length = target_kb.concept_len(hypotheses[0].concept)
            concept_inds = concept_ind.intersection(
                positive_examples_test | negative_examples_test
            )

            all_examples = positive_examples_test.union(negative_examples_test)
            predicitons_dict = {
                item: 1 if item in concept_inds else 0 for item in all_examples
            }
            explanation_dict[str_target_concept] = {
                "best_concept": str(best_concept),
                "concept_length": concept_length,
            }
            predictions_data[str_target_concept] = {
                "concept_individuals": concept_inds,
                "positive_examples_test": examples["positive_examples_test"],
                "negative_examples_test": examples["negative_examples_test"]
            }

        t1 = time.time()
        duration = t1 - t0
        metrics = calculate_metrics(predictions_data)
        print("metrics")
        print(metrics)
        return predictions_data, duration, explanation_dict, metrics


def create_learning_problems_from_csv(csv_path: str, test_size=1, random_state=42):
    """
    Reads a predictions CSV and creates learning problems for Ontolearn.
    The CSV must contain 'uri' and 'predicted_label' columns.
    """
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['uri', 'predicted_label']):
             print("Error: CSV must contain 'uri' and 'predicted_label' columns.")
             return None
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please create this file.")
        return None

    learning_problems = {}
    pred_df = pd.read_csv("predictions/predictions.csv")
    pos_test = pred_df[pred_df["predicted_label"] == 1].copy()
    neg_test = pred_df[pred_df["predicted_label"] == 0].copy()

    # --- Load and split train data -------------------------------------------
    train_df = pd.read_csv("data/trainingSet.tsv", sep="\t")  # columns: id, uri, label
    train_df = train_df.drop(columns="id")
    pos_train = train_df[train_df["label"] == 1].copy()
    neg_train = train_df[train_df["label"] == 0].copy()

    # --- Store learning problems by label ------------------------------------
    learning_problems = {
        1: {
            "positive_examples_train": pos_train,
            "negative_examples_train": neg_train,
            "positive_examples_test": pos_test,
            "negative_examples_test": neg_test,
        },
        0: {
            "positive_examples_train": neg_train,
            "negative_examples_train": pos_train,
            "positive_examples_test": neg_test,
            "negative_examples_test": pos_test,
        },
    }
    return learning_problems


def EvoLearnerResult():
    # --- Configuration ---
    # Define the path to your predictions file and the name of your Knowledge Graph
    PREDICTIONS_FILE = "predictions/predictions.csv"
    # This should be the name of your .owl file (without the extension)
    # located in the 'data/KGs/' directory.
    KG_NAME = "family"

    print("Starting EvoLearner...")
    
    # --- Setup ---

    kg_path = f"data/{KG_NAME}.owl"
    
    if not os.path.exists(kg_path):
        print(f"Warning: Knowledge Graph '{kg_path}' not found.")

    if not os.path.exists(PREDICTIONS_FILE):
        print(f"Warning: Predictions file '{PREDICTIONS_FILE}' not found.")
    # --- End of Setup ---

    # 1. Create learning problems from the predictions file
    print("Step 1: Creating learning problems from CSV...")
    learning_problems = create_learning_problems_from_csv(PREDICTIONS_FILE)

    if learning_problems:
        print(f"Successfully created learning problems for labels: {list(learning_problems.keys())}")
        
        # 2. Run EvoLearner to find explanatory concepts
        print("\nStep 2: Running EvoLearner to find explanations...")
        results = train_evo(learning_problems, kg=KG_NAME)
        
        predictions, duration, explanations, metrics = results
        return predictions, duration, explanations, metrics
        

