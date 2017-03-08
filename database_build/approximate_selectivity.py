import sys, re, os.path
from sparql_graph.query_vector_converter import jena_graph
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
		with open(os.path.dirname(__file__) + '/dbpedia_stats/property_triple_count.txt') as f:
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
		return sum(selectivity)
		
	else:
		return 0.0

def main():
	print get_selectivity('DESCRIBE <http://dbpedia.org/resource/Beaudesert,_Queensland>')

if __name__ == '__main__':
	main()