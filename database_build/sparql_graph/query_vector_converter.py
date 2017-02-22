'''
Executes Main.java in the same directory and prints it out
'''
import subprocess, re, os.path, sys
from subprocess import STDOUT,PIPE,Popen
import pyparsing as pp
from pprint import pprint
from collections import defaultdict


# def compile_java(java_file):
# 	print "FUCK"
# 	cmd = ["javac", "-classpath", '/Users/David/libs/jena/lib/*:.', java_file]
# 	proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE)
# 	stdout, stderr = proc.communicate()

def convert_query(query):
	result = -1

	# if not (os.path.isfile('Main.class')):
	# 	compile_java('Main.java')
	query = re.sub(r'([A-Z]{3,})', r' \1', query)
	# query = re.sub(r'"', r'\\\1', query)
	query = ' '.join(query.split())
	try:
		tree = jena_graph('Main', query)
		result = gen_graph(tree[0])

	except:
		print "gen_graph err", sys.exc_info()[0]
	return result

# executes Main.java in same folder to convert to SPARQL Algebra expressino
def jena_graph(java_file, args):
	graph = ''
	#Makes the call to start Main.java - gets output of file via System.out.println(string)
	cmd = ["java", "-classpath", "/Users/David/libs/jena/lib/*:.", java_file, args]
	proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
	stdout, stderr = proc.communicate()

	for line in stdout:
		# print line,
		graph += line

	print graph
	try:
		graph = pp.nestedExpr(opener='(', closer=')').parseString(graph)

		return graph.asList()
	except ParseException as pe:
		print "parse exception", pe
		print "column: {}".format(pe.col)
	except:
		print "pyparsing nestedExpr error", sys.exc_info()[0]

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
	query_cmds = {'select': [0,0,0], 'from':[0,0,0], 'union': [0,0,0], 'prefix':[0,0,0], 'bgp':[0,0,0], 'triple':[0,0,0], 'describe': [0,0,0]}
	tokenized_graph_list = list(tokenize_list(tree))
	for val,depth in tokenized_graph_list:
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

	query_list = []
	for key in sorted(query_cmds):
		query_list = query_list + query_cmds[key]
	return query_list



# def main():
# 	# compile_java('Main.java')
# 	print convert_query("CONSTRUCT { <http://dbpedia.org/resource/Diane_Fletcher> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x1 . <http://dbpedia.org/resource/Diane_Fletcher> <http://www.w3.org/2000/01/rdf-schema#label> ?x2 . }WHERE { { <http://dbpedia.org/resource/Diane_Fletcher> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x1 . } UNION { <http://dbpedia.org/resource/Diane_Fletcher> <http://www.w3.org/2000/01/rdf-schema#label> ?x2 . } }")


# if __name__ == '__main__':
# 	main()
