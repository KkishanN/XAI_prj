from SGC import train_and_evaluate_sgc
from SAGE import run_uncle_classifier
import pandas as pd
import sys
import argparse
from tabulate import tabulate
from EvoLearner import EvoLearnerResult
from pathlib import Path

def process_metrics(metrics):
    """Process metrics dictionary into a formatted DataFrame row."""
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df = metrics_df.rename(columns={
        'accuracy': 'Exp Accuracy',
        'precision': 'Exp Precision', 
        'recall': 'Exp Recall',
        'f1-score': 'Exp F1'
    })
    metrics_row = {}
    for concept in sorted(metrics.keys()):
        for metric in ['Exp Accuracy', 'Exp Precision', 'Exp Recall', 'Exp F1']:
            metrics_row[f"{metric} (Concept {concept})"] = metrics_df.loc[concept, metric]
    
    return pd.DataFrame([metrics_row])

def create_results_df(model_name, val_metrics):
    """Create a results DataFrame for a given model."""
    return pd.DataFrame({
        'Model': [model_name.upper()],
        'Accuracy': [f"{val_metrics['accuracy']:.4f}"],
        'Precision': [f"{val_metrics['precision']:.4f}"],
        'Recall': [f"{val_metrics['recall']:.4f}"],
        'F1': [f"{val_metrics['f1']:.4f}"]
    })

def run_sgc_model():
    """Run the SGC model and return processed results."""
    print("Running SGC model...")
    sgc_result = train_and_evaluate_sgc()
    val_metrics = sgc_result['metrics']
    results_df = create_results_df('sgc', val_metrics)
    
    try:
        _, _, _, metrics = EvoLearnerResult()
        metrics_df = process_metrics(metrics)
        final_df = pd.concat([results_df, metrics_df], axis=1)
        final_df.to_csv('results/sgc_res.csv', index=False)
        print("SGC results saved successfully.")
        return final_df
    except Exception as e:
        print(f"Error processing SGC explanations: {e}")
        return results_df

def run_sage_model():
    """Run the SAGE model and return processed results."""
    print("Running SAGE model...")
    sage_result = run_uncle_classifier()
    val_metrics = sage_result['val_metrics']
    results_df = create_results_df('sage', val_metrics)
    
    try:
        _, _, _, metrics = EvoLearnerResult()
        metrics_df = process_metrics(metrics)
        final_df = pd.concat([results_df, metrics_df], axis=1)
        final_df.to_csv('results/sage_res.csv', index=False)
        print("SAGE results saved successfully.")
        return final_df
    except Exception as e:
        print(f"Error processing SAGE explanations: {e}")
        return results_df

def show_results():
    """Display combined results from both models."""
    try:
        # Load results with error handling for missing columns
        sage_df = pd.read_csv('results/sage_res.csv')
        sgc_df = pd.read_csv('results/sgc_res.csv')
        
        # Define columns to display (prioritizing Concept 1 if multiple exist)
        display_columns = [
            'Model', 'Accuracy', 'Precision', 'Recall', 'F1',
            'Exp Accuracy (Concept 1)', 'Exp Precision (Concept 1)',
            'Exp Recall (Concept 1)', 'Exp F1 (Concept 1)'
        ]
        
        # Filter and rename columns
        column_mapping = {
            'Model': 'Model',
            'Accuracy': 'Pred Accuracy',
            'Precision': 'Pred Precision',
            'Recall': 'Pred Recall',
            'F1': 'Pred F1',
            'Exp Accuracy (Concept 1)': 'Exp Accuracy',
            'Exp Precision (Concept 1)': 'Exp Precision',
            'Exp Recall (Concept 1)': 'Exp Recall',
            'Exp F1 (Concept 1)': 'Exp F1'
        }
        
        sage_df = sage_df[display_columns].rename(columns=column_mapping)
        sgc_df = sgc_df[display_columns].rename(columns=column_mapping)
        
        combined_df = pd.concat([sage_df, sgc_df], ignore_index=True)
        
        print("\nModel Performance Comparison:")
        print(tabulate(combined_df, headers='keys', tablefmt='grid', 
                      showindex=False, floatfmt=".4f"))
        
    except FileNotFoundError:
        print("Error: Results files not found. Please run models first.")
    except KeyError as e:
        print(f"Error: Missing expected column in results - {e}. Please check your data.")
    except Exception as e:
        print(f"Unexpected error showing results: {e}")

def main():
    # Ensure results directory exists
    Path('results').mkdir(exist_ok=True)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Graph Classification Model Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-model', type=str, 
                       choices=['sgc', 'sage'],
                       help='Model to run (SGC or SAGE)')
    parser.add_argument('-show_results', action='store_true',
                       help='Display comparison of saved results')
    args = parser.parse_args()
    
    # Execute requested actions
    if args.model:
        try:
            if args.model.lower() == 'sgc':
                run_sgc_model()
            elif args.model.lower() == 'sage':
                run_sage_model()
        except Exception as e:
            print(f"Error running {args.model.upper()} model: {e}")
    
    if args.show_results:
        show_results()
    
    if not args.model and not args.show_results:
        parser.print_help()

if __name__ == "__main__":
    main()