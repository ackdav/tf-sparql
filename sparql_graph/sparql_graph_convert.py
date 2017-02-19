'''
Executes Main.java in the same directory and prints it out
'''
import subprocess, re
from subprocess import STDOUT,PIPE

# def compile_java(java_file):
# 	cmd = ["javac", "-classpath", '.:/Users/David/libs/jena/lib/*' + java_file]
# 	proc = subprocess.Popen(cmd)

def execute_java(java_file, stdin):
	graph = ''
	p = subprocess.Popen(["java", "-classpath", ".:/Users/David/libs/jena/lib/*", "Main", stdin], stdout=subprocess.PIPE)
	for line in iter(p.stdout.readline, b''):
		# print line,
		graph += line
	return graph

def traverse_graph(tree):
	for line in iter(tree):
		if line.count("(")>0:
			print line

def main():
	query = 'PREFIX dbpedia-de: <http://de.dbpedia.org/resource/>PREFIX dbpedia-owl: <http://dbpedia.org/ontology/>PREFIX dbpedia-fr: <http://fr.dbpedia.org/resource/>PREFIX owl: <http://www.w3.org/2002/07/owl#>PREFIX dcterms: <http://purl.org/dc/terms/>PREFIX dbpedia: <http://dbpedia.org/resource/>PREFIX dbpprop: <http://dbpedia.org/property/>PREFIX foaf: <http://xmlns.com/foaf/0.1/>SELECT ?resWHERE { dbpedia:Rejection_sampling dbpedia-owl:abstract ?res FILTER langMatches(lang(?res), "en") }'
	query = re.sub(r'([A-Z]{3,})', r' \1', query)

	tree = execute_java('Main.java', query)
	traverse_graph(tree)


if __name__ == '__main__':
	main()