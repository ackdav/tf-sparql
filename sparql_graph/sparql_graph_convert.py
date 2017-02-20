'''
Executes Main.java in the same directory and prints it out
'''
import subprocess, re
from subprocess import STDOUT,PIPE,Popen
import pyparsing as pp

def compile_java(java_file):
	cmd = ["javac", "-classpath", '/Users/David/libs/jena/lib/*:.', java_file]
	proc = subprocess.Popen(cmd)

# executes Main.java in same folder to convert to SPARQL Algebra expressino
def execute_java(java_file, args):
	graph = ''
	cmd = ["java", "-classpath", "/Users/David/libs/jena/lib/*:.", java_file, args]
	proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
	stdout, stderr = proc.communicate()

	for line in stdout:
		graph += line

	print graph
	return graph


def parse_sexp(string):
    """
    >>> parse_sexp("(+ 5 (+ 3 5))")
    [['+', '5', ['+', '3', '5']]]
    
    """
    sexp = [[]]
    word = ''
    in_str = False
    for char in string:
        if char == '(' and not in_str:
            sexp.append([])
        elif char == ')' and not in_str:
            if word:
                sexp[-1].append(word)
                word = ''
            temp = sexp.pop()
            sexp[-1].append(temp)
        elif char in (' ', '\n', '\t') and not in_str:
            if word:
                sexp[-1].append(word)
                word = ''
        elif char == '\"':
            in_str = not in_str
        else:
            word += char
    return sexp[0]

# def traverse_graph(tree):
	
# 	LP = pp.Literal("(").suppress()
# 	RP = pp.Literal(")").suppress()
# 	String = pp.Word(pp.alphanums + '_')
# 	SingleQuoteString = pp.QuotedString(quoteChar="'", unquoteResults=False)
# 	DoubleQuoteString = pp.QuotedString(quoteChar='"', unquoteResults=False)
# 	QuotedString = SingleQuoteString | DoubleQuoteString
# 	Atom = String | QuotedString
# 	SExpr = pp.Forward()
# 	SExprList = pp.Group(pp.ZeroOrMore(SExpr | Atom))
# 	SExpr << (LP + SExprList + RP)

# 	data = '(gimme [some {nested, nested [lists]}])' 

# 	print enclosed.parseString(tree).asList()

def main():
	compile_java('Main.java')
	query = "CONSTRUCT { <http://dbpedia.org/resource/Diane_Fletcher> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x1 . <http://dbpedia.org/resource/Diane_Fletcher> <http://www.w3.org/2000/01/rdf-schema#label> ?x2 . }WHERE { { <http://dbpedia.org/resource/Diane_Fletcher> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x1 . } UNION { <http://dbpedia.org/resource/Diane_Fletcher> <http://www.w3.org/2000/01/rdf-schema#label> ?x2 . } }"
	query = re.sub(r'([A-Z]{3,})', r' \1', query)
	query = ' '.join(query.split())


	tree = execute_java('Main', query)
	root = parse_sexp(tree)
	print root

if __name__ == '__main__':
	main()
