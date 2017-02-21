'''
Executes Main.java in the same directory and prints it out
'''
import subprocess, re
from subprocess import STDOUT,PIPE,Popen
import pyparsing as pp
from pprint import pprint
from collections import defaultdict


def compile_java(java_file):
	cmd = ["javac", "-classpath", '/Users/David/libs/jena/lib/*:.', java_file]
	proc = subprocess.Popen(cmd)

# executes Main.java in same folder to convert to SPARQL Algebra expressino
def execute_java(java_file, args, nested_list_convert):
	graph = ''

	#Makes the call to start Main.java - gets output of file via System.out.println(string)
	cmd = ["java", "-classpath", "/Users/David/libs/jena/lib/*:.", java_file, args]
	proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
	stdout, stderr = proc.communicate()

	for line in stdout:
		graph += line

	# print graph

	#Converts graph to a nested list
	if nested_list_convert == True:
		graph = pp.nestedExpr(opener='(', closer=')').parseString(graph)
		return graph.asList()
	else:
	   return graph

def tokenize_list(nested_list, d=0):
    if not isinstance(nested_list, list):
        yield nested_list, d
    else:
        for item in nested_list:
            for x in tokenize_list(item, d=d+1):
            	# if x[0][0]!='?' and x[0][0]!='<' and x[0]!='triple':
            	if x[0][0]!='?' and x[0][0]!='<':
                	yield x
    return


def gen_graph(tree):
	'''
	query_cmds def per index:
	1. number of occurences
	2. min height in graph
	3. max height in graph
	'''
	query_cmds = {'select': [0,0,0], 'from':[0,0,0], 'union': [0,0,0], 'prefix':[0,0,0], 'bgp':[0,0,0], 'triple':[0,0,0]}
	token_l = list(tokenize_list(tree))
	for val,depth in token_l:
		if val.lower() in query_cmds:
			# occurence + 1
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
	for key in sorted(query_cmds):
		print "%s: %s" % (key, query_cmds[key])

def main():
	compile_java('Main.java')
	query = "CONSTRUCT { <http://dbpedia.org/resource/Diane_Fletcher> ?p1 ?x2 . <http://dbpedia.org/resource/Diane_Fletcher> <http://dbpedia.org/ontology/genre> ?x3 . ?x3 <http://www.w3.org/2000/01/rdf-schema#label> ?x4 . <http://dbpedia.org/resource/Diane_Fletcher> <http://dbpedia.org/ontology/party> ?x5 . ?x5 <http://www.w3.org/2000/01/rdf-schema#label> ?x6 . <http://dbpedia.org/resource/Diane_Fletcher> <http://dbpedia.org/ontology/birthPlace> ?x7 . ?x7 <http://www.w3.org/2000/01/rdf-schema#label> ?x8 . }WHERE { { <http://dbpedia.org/resource/Diane_Fletcher> ?p1 ?x2 . } UNION { {<http://dbpedia.org/resource/Diane_Fletcher> <http://dbpedia.org/ontology/genre> ?x3 . ?x3 <http://www.w3.org/2000/01/rdf-schema#label> ?x4 . } } UNION { {<http://dbpedia.org/resource/Diane_Fletcher> <http://dbpedia.org/ontology/party> ?x5 . ?x5 <http://www.w3.org/2000/01/rdf-schema#label> ?x6 . } } UNION { {<http://dbpedia.org/resource/Diane_Fletcher> <http://dbpedia.org/ontology/birthPlace> ?x7 . ?x7 <http://www.w3.org/2000/01/rdf-schema#label> ?x8 . } } }"
	query = re.sub(r'([A-Z]{3,})', r' \1', query)
	query = ' '.join(query.split())

	tree = execute_java('Main', query, True)

	gen_graph(tree[0])


if __name__ == '__main__':
	main()
