'''
Executes Main.java in the same directory and prints it out
'''
import subprocess, re, os.path, sys, collections
from subprocess import STDOUT,PIPE,Popen
import pyparsing as pp
from pprint import pprint
from collections import defaultdict


# def compile_java(java_file):
# 	cmd = ["javac", "-classpath", '/Users/David/libs/jena/lib/*:.', java_file]
# 	proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE)
# 	stdout, stderr = proc.communicate()
# executes Main.java in same folder to convert to SPARQL Algebra expressino

def convert_query(query):
	result = -1

	query = re.sub(r'([A-Z]{3,})', r' \1', query)
	query = ' '.join(query.split())

	try:
		tree = jena_graph('Main', query)
		result = gen_graph(tree[0])
	except:
		print "gen_graph err", sys.exc_info()[0]
	return result

def jena_graph(java_file, args):
	'''
	Starts Main.java in the same folder and converts it into a query tree, then into a nested list
	'''

	graph = ''
	#Makes the call to start Main.java - gets output of file via System.out.println(string)
	cmd = ["java", "-classpath", "/Users/David/libs/jena/lib/*:.", java_file, args]
	proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
	stdout, stderr = proc.communicate()

	for line in stdout:
		graph += line
	print graph

	try:
		# TODO: escape URLs containing '(' and ')'
		graph = pp.nestedExpr(opener='(', closer=')').parseString(graph)

		return graph.asList()

	except ParseException as pe:
		print "parse exception", pe
		print "column: {}".format(pe.col)
	except:
		print "pyparsing nestedExpr error", sys.exc_info()[0]

# def subject_selectivity(triple):
# 	db_triple_num_s = {}
# 	total_triples = 458315769

# 	with open('db-triple-subject-count.txt') as f:
# 		for line in f:
# 			line = line.split('\t')
# 			db_triple_num_s[line[1].strip('\n')] = line[0]

# 	if triple[1][0] != '?':
# 		if triple[1][1:-1] in db_triple_num_s:
# 			print 'XXXXX'
# 			print ('subject_selectivity', float(db_triple_num_s[triple[1][1:-1]]/total_triples))

# 	else:
# 		return ('subject_selectivity', -1)


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
		# print nested_list
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


def gen_graph(tree):
	'''
	query_cmds def per index:
	1. number of occurences
	2. min height in graph
	3. max height in graph
	'''
	maxdepth = 0
	query_cmds = {'maxdepth': [0], 'wordcount': [0],\
					 'union': [0,0,0], 'project':[0,0,0], 'bgp':[0,0,0], 'triple':[0,0,0], \
						'distinct': [0,0,0], 'order': [0,0,0], 'leftjoin':[0,0,0], 'filter': [0,0,0],\
					 		'bgptype':[0,0,0,0,0,0,0,0]}

	tokenized_graph_list = list(tokenize_list(tree))
	query_cmds['wordcount'][0] = len(tokenized_graph_list)

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
	print convert_query(" PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?type WHERE { <http://dbpedia.org/resource/Cityvibe> rdf:type ?type }")


if __name__ == '__main__':
	main()
