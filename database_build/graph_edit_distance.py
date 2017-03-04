import  sys
from subprocess import STDOUT,PIPE,Popen

def get_distances(query, query_list):
    distances = []
    for benchmark_query in query_list:
    	try:
            cmd = ["/Users/David/Documents/ged-wrap/scripts/qdistance-beam", "--std", query, benchmark_query, "--beam 500"]
            print cmd
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
            stdout, stderr = proc.communicate()
            print stdout
            if not 'help' in stdout:
                distances.append(float(stdout))
        except:
			print "GED err", sys.exc_info()[0], query
    return distances

def main():
    C = "CONSTRUCT { <http://dbpedia.org/resource/Saif_Salman> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x1 . <http://dbpedia.org/resource/Saif_Salman> <http://www.w3.org/2000/01/rdf-schema#label> ?x2 . }WHERE { { <http://dbpedia.org/resource/Saif_Salman> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?x1 . } UNION { <http://dbpedia.org/resource/Saif_Salman> <http://www.w3.org/2000/01/rdf-schema#label> ?x2 . } }"
    D = ["CONSTRUCT { <http://dbpedia.org/resource/Saif_Salman> ?p1 ?x2 . <http://dbpedia.org/resource/Saif_Salman> <http://dbpedia.org/ontology/genre> ?x3 . ?x3 <http://www.w3.org/2000/01/rdf-schema#label> ?x4 . <http://dbpedia.org/resource/Saif_Salman> <http://dbpedia.org/ontology/party> ?x5 . ?x5 <http://www.w3.org/2000/01/rdf-schema#label> ?x6 . <http://dbpedia.org/resource/Saif_Salman> <http://dbpedia.org/ontology/birthPlace> ?x7 . ?x7 <http://www.w3.org/2000/01/rdf-schema#label> ?x8 . }WHERE { { <http://dbpedia.org/resource/Saif_Salman> ?p1 ?x2 . } UNION { {<http://dbpedia.org/resource/Saif_Salman> <http://dbpedia.org/ontology/genre> ?x3 . ?x3 <http://www.w3.org/2000/01/rdf-schema#label> ?x4 . } } UNION { {<http://dbpedia.org/resource/Saif_Salman> <http://dbpedia.org/ontology/party> ?x5 . ?x5 <http://www.w3.org/2000/01/rdf-schema#label> ?x6 . } } UNION { {<http://dbpedia.org/resource/Saif_Salman> <http://dbpedia.org/ontology/birthPlace> ?x7 . ?x7 <http://www.w3.org/2000/01/rdf-schema#label> ?x8 . } } }"]
    print get_distances(C, D)

if __name__ == '__main__':
    main()