import sys, re, os.path
from query_vector_converter import jena_graph
from graph_edit_distance import rewrite_describe_queries

def extract_list(nested_list, d=0):
	if isinstance(nested_list, list):
		if nested_list[0] == 'triple':
			yield  nested_list
		for item in nested_list:
			for x in extract_list(item, d=d+1):
				yield x
	return

def extract_triples(query):
	nested_list = jena_graph('Main', query)
	
	triples =  list(extract_list(nested_list))

	return triples

def subject_selectivity(triple):

	if '?' in triple[1]:
		return 1.0
	else:
		#TODO
		return 1.0

def property_selectivity(triple):

	if '?' in triple[2]:
		return 1.0
	else:
		total = 0.
		occurences = 0.
		with open(os.path.dirname(os.path.abspath(__file__))+ '/property_triple_count.txt') as f:
			for line in f:
				line = line.split('\t')
				total += float(line[1])
				if line[0] == triple[2][1:-1]:
					occurences = float(line[1])
		return float(occurences/total)

def get_selectivity(query):
	if 'DESCRIBE' in query:
		query = rewrite_describe_queries(query)
		
	triples = extract_triples(query)
	selectivity = []
	if len(triples) > 0:
		for t in triples:
			s_select = subject_selectivity(t)
			p_select = property_selectivity(t)
			selectivity.append(s_select * p_select)
		prod = 1
		for x in selectivity:
			prod += x
		return prod
		
	else:
		return 0.0

def main():
	print get_selectivity('PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> select ?label ?abstract ?thumbnail where {?resource rdfs:label ?label ; dbpedia-owl:abstract ?abstract . OPTIONAL { ?resource dbpedia-owl:thumbnail ?thumbnail } FILTER( ?resource = <http://dbpedia.org/resource/Human_factors_and_ergonomics> || ?resource = <http://dbpedia.org/resource/Human_Factors_and_Ergonomics_Society> || ?resource = <http://dbpedia.org/resource/The_Institute_of_Ergonomics_and_Human_Factors> || ?resource = <http://dbpedia.org/resource/Cryptography> || ?resource = <http://dbpedia.org/resource/Proceedings_of_the_Human_Factors_and_Ergonomics_Society_Annual_Meeting> || ?resource = <http://dbpedia.org/resource/Human_Factors_(journal)> || ?resource = <http://dbpedia.org/resource/Human_reliability> || ?resource = <http://dbpedia.org/resource/Ergonomics_in_Design> || ?resource = <http://dbpedia.org/resource/High-velocity_human_factors> || ?resource = <http://dbpedia.org/resource/Engineering_psychology> ) filter(langMatches(lang(?abstract),\"en\")) filter(langMatches(lang(?label),\"en\"))}')

if __name__ == '__main__':
	main()
