'''
Executes Main.java in the same directory and prints it out
'''
import subprocess, re, sys, collections, os, time
from subprocess import STDOUT,PIPE,Popen
import pyparsing as pp
from collections import defaultdict
from graph_edit_distance import rewrite_describe_queries

# def compile_java(java_file):
# 	cmd = ["javac", "-classpath", '/Users/David/libs/jena/lib/*:.', java_file]
# 	proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE)
# 	stdout, stderr = proc.communicate()
# executes Main.java in same folder to convert to SPARQL Algebra expressino

# def clean_query(query):
# 	query = re.sub(r'([A-Z]{3,})', r' \1', query)
# 	query = ' '.join(query.split())
# 	return query

def structural_query_vector(query):
	result = -1
	
	if ('DESCRIBE') in query:
		query = rewrite_describe_queries(query)
	try:
		result = jena_graph('Main', query)
		result = algebra_structure_feature_vector(result[0])
	except:
		print "algebra_structure_feature_vector err", sys.exc_info()[0]
		return -1
	return result

def jena_graph(java_file, args):
	'''
	Starts Main.java in the same folder and converts it into a query tree, then into a nested list
	'''
	graph = ''
	#Makes the call to start Main.java - gets output of file via System.out.println(string)
	cmd = ["java", "-cp", os.environ.get('CLASSPATH'), java_file, args]
	proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
	stdout, stderr = proc.communicate()

	for line in stdout:
		graph += line
	
	try:
		res_graph = pp.nestedExpr(opener='(', closer=')').parseString(graph)
	
		res_graph = res_graph.asList()
	except:
		print "pyparse err", graph, args
		res_graph = -1
	return res_graph


def bgp_type_match(bgp_list):
	'''Matching to one of 8 BGP types:
	1. ?s o p 	5. ?s o ?p
	2. s ?o p 	6. s ?o ?p
	3. s o ?p 	7. ?s ?o ?p
	4. ?s ?o p 	8. s o p
	'''

	if bgp_list[1][0] == '?':
		if bgp_list[2][0] == '?':
			if bgp_list[3][0] == '?':
				return ('bgptype', 7)
			else:
				return ('bgptype', 4)
		elif bgp_list[3][0] == '?':
			return ('bgptype', 5)
		else:
			return ('bgptype', 1)
	elif bgp_list[2][0] == '?':
		if bgp_list[3][0] == '?':
			return ('bgptype', 6)
		else:
			return ('bgptype', 2)
	elif bgp_list[3][0] == '?':
		return ('bgptype', 3)
	else:
		return ('bgptype', 8)

def tokenize_list(nested_list, d=0):
	if not isinstance(nested_list, list):
		yield nested_list, d
	else:
		if nested_list[0] == 'triple':
			# !!! second value of bgptype represents type, and not depth
			yield  bgp_type_match(nested_list)
		for item in nested_list:
			for x in tokenize_list(item, d=d+1):
				# if x[0][0]!='?' and x[0][0]!='<' and x[0]!='triple':
				# if x[0][0]!='?' and x[0][0]!='<' and x[0][0]!='>':
				yield x
	return


def algebra_structure_feature_vector(tree):
	'''
	query_cmds def per index:
	1. number of occurences
	2. min height in graph
	3. max height in graph
	'''
	maxdepth = 0
	query_cmds = {'maxdepth': [0], 'wordcount': [0],\
					 'triple': [0,0,0], 'bgp':[0,0,0], 'leftjoin':[0,0,0], 'union':[0,0,0], \
						'tolist': [0,0,0], 'order': [0,0,0], 'project':[0,0,0], 'distinct': [0,0,0],\
						'reduced': [0,0,0], 'multi': [0,0,0], 'top':[0,0,0], 'group': [0,0,0], \
							'assign': [0,0,0], 'sequence':[0,0,0], 'bgptype':[0,0,0,0,0,0,0,0] }

	tokenized_graph_list = list(tokenize_list(tree))
	query_cmds['wordcount'][0] = len(tokenized_graph_list)
	# print "LENGTH", len(tokenized_graph_list)
	# TEST: test structural feature effectiveness
	# if len(tokenized_graph_list) <= 8:
	# 	return -1
	for val, depth in tokenized_graph_list:
		if depth > maxdepth:
			maxdepth = depth
		if val.lower() in query_cmds:
			#handle bgptypes - depth == bgptype here and not depth
			if val.lower() == 'bgptype':
				query_cmds[val.lower()][depth-1] += 1
			else:
				#handle rest
				query_cmds[val.lower()][0] += 1
				#if new entry - initialize min and max depth
				if query_cmds[val.lower()][0] == 1:
					query_cmds[val.lower()][1] = 1
					query_cmds[val.lower()][2] = 1
				else:
					if query_cmds[val.lower()][1] > depth:
						query_cmds[val.lower()][1] = depth
					if query_cmds[val.lower()][2] < depth:
						query_cmds[val.lower()][2] = depth
	query_list = []
	for key in sorted(query_cmds):
		query_list = query_list + query_cmds[key]
	return query_list



def main():
	# compile_java('Main.java')
	# escaped = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>SELECT DISTINCT ?anton_tchekov ?anton_tchekov_field_auteurWHERE {?anton_tchekov rdfs:label "Anton Tchekov"@en; rdfs:label ?anton_tchekov_field_auteur. } LIMIT 5".replace('"','\\"')
	# print escaped
	print structural_query_vector("CONSTRUCT  { ?subject ?predicate ?object } WHERE { ?subject ?predicate ?object . FILTER (  ?subject = <http://dbpedia.org/resource/ DBL-583> || ?object = <http://dbpedia.org/resource/ DBL-583>) }")


if __name__ == '__main__':
	main()
