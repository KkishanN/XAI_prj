# This files verifies if the conversion from owl to .nt tripes was correct
import rdflib
import sys

def verify_triple_count(original_file, converted_file):
    
    g_original = rdflib.Graph()
    g_converted = rdflib.Graph()

    try:
        # Load the original owl file
        print(f"Parsing original file: {original_file} ...")
        g_original.parse(original_file, format='xml')
        original_count = len(g_original)

        # Load the converted N-Triples file
        print(f"Parsing converted file: {converted_file} ...")
        g_converted.parse(converted_file, format='nt')
        converted_count = len(g_converted)

        # Compare the counts
        print("\n--- Verification Result ---")
        if original_count == converted_count:
            print(f"SUCCESS: The number of triples is identical ({original_count}). The conversion was successful.")
        else:
            print(f"FAILURE: Triple counts do not match!")
            print(f"Original file has {original_count} triples.")
            print(f"Converted file has {converted_count} triples.")


        from rdflib.compare import isomorphic

        # Compare the graphs for semantic equivalence
        print("\n--- Semantic Equivalence Check ---")
        if isomorphic(g_original, g_converted):
            print("SUCCESS: The graphs are semantically identical (isomorphic).")
        else:
            print("FAILURE: The graphs are NOT isomorphic.")

        
        return original_count == converted_count

    except Exception as e:
        print(f"\nAn error occurred during verification: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) == 3:
        original_rdf_file = sys.argv[1]
        converted_nt_file = sys.argv[2]
        # print("Usage: python verify_conversion.py <originalFile.rdf> <convertedFile.nt>")
    else:
        # Default files
        original_rdf_file = 'Family/family.owl'  # Your original .rdf or .owl file
        converted_nt_file = 'data/family.nt'   # The .nt file you created


    verify_triple_count(original_rdf_file, converted_nt_file)
