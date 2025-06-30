import pandas as pd
from rdflib import Graph, Namespace, RDF

def family_node_features():
    # Load the complete dataset
    df = pd.read_csv("data/completeDataset.tsv", sep="\t")
    ground_truth = {row['uri']: row['label'] for _, row in df.iterrows()}

    g = Graph()
    g.parse("data/family.owl", format="xml")
    family_ns = Namespace("http://www.benchmark.org/family#")

    # dictionaries to store relationships
    sibling_map = {}
    spouse_map = {}
    has_child_map = {}

    # Collect all family individuals
    family_uris = [str(uri) for uri in g.subjects(RDF.type, None) 
                if str(uri).startswith(str(family_ns))]

    # relationship maps
    for uri in family_uris:
        uri_ref = family_ns[uri.split("#")[-1]]
        
        # Siblings
        siblings = set()
        for sib in g.objects(uri_ref, family_ns.hasSibling):
            siblings.add(str(sib))
        for sib in g.subjects(family_ns.hasSibling, uri_ref):
            siblings.add(str(sib))
        sibling_map[uri] = siblings
        
        # Spouses
        spouses = set()
        for spouse in g.objects(uri_ref, family_ns.married):
            spouses.add(str(spouse))
        for spouse in g.subjects(family_ns.married, uri_ref):
            spouses.add(str(spouse))
        spouse_map[uri] = spouses
        
        # Children
        has_child_map[uri] = any(g.triples((uri_ref, family_ns.hasChild, None)))

    #features for each URI in ground truth
    results = []
    for uri, true_label in ground_truth.items():
        if uri not in family_uris:
            continue
            
        # Basic features
        is_male = (family_ns[uri.split("#")[-1]], RDF.type, family_ns.Male) in g
        has_sibling = bool(sibling_map.get(uri, set()))
        
        # Sibling has child
        sibling_has_child = False
        for sibling_uri in sibling_map.get(uri, set()):
            if sibling_uri in has_child_map and has_child_map[sibling_uri]:
                sibling_has_child = True
                break
        
        # Married to someone with siblings
        married_to_sibling = False
        for spouse_uri in spouse_map.get(uri, set()):
            if sibling_map.get(spouse_uri, set()):
                married_to_sibling = True
                break
        
        our_prediction = 1 if is_male and (sibling_has_child or married_to_sibling) else 0
        
        match = 1 if our_prediction == true_label else 0
        
        # Add to results (using 1/0)
        person_id = uri.split("#")[-1]
        results.append({
            "ID": person_id,
            "Male": 1 if is_male else 0,
            "Has_Sibling": 1 if has_sibling else 0,
            "Sibling_Has_Child": 1 if sibling_has_child else 0,
            "Married_to_Sibling_Holder": 1 if married_to_sibling else 0,
            "Uncle": our_prediction  # This will serve as our label
        })

    # DataFrame
    results_df = pd.DataFrame(results)
    return results_df
