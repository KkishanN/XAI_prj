
import rdflib
import sys

def convert_owl_to_rdf(input_file, output_file):
    """
    Converts an OWL file to an RDF file.
    """
    # Create a new RDF graph
    g = rdflib.Graph()

    try:
        # Load the input OWL file. The 'xml' parser can handle OWL/XML.
        print(f"Loading and parsing {input_file}...")
        g.parse(input_file, format='xml')
        print("Parsing complete.")

        # Serialize the graph to the output file in N-triples format.
        print(f"Saving to {output_file} in N-triples format...")
        g.serialize(destination=output_file, format='nt')
        print(f"Successfully converted {input_file} to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        input_owl_file = sys.argv[1]
        output_rdf_file = sys.argv[2]
    else:
        # Default files
        input_owl_file = 'data/family.owl'
        output_rdf_file = 'data/family.nt'
        print("Usage: python convert_single.py <inputFile.owl> <outputFile.rdf>")
        print(f"Using default files: {input_owl_file} -> {output_rdf_file}\n")

    convert_owl_to_rdf(input_owl_file, output_rdf_file)
