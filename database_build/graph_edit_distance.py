import  sys
from subprocess import STDOUT,PIPE,Popen

def add_missing_prefixes(query):
    with open('jena-missing-prefixes.txt') as f:
        for prefix in f:
            query = prefix + " " + query
    return query

def get_distances(query, query_list):
    distances = []
    query = add_missing_prefixes(query)
    for benchmark_query in query_list:
    	try:
            benchmark_query = add_missing_prefixes(benchmark_query)
            cmd = ["/Users/David/Documents/ged-wrap/scripts/qdistance-beam --std \"%s\" \"%s\" --beam 500" % (query, benchmark_query)]
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
            stdout, stderr = proc.communicate()
            if not 'help' in stdout:
                distances.append(float(stdout))
        except:
            distances.append(-1.0)
            # print "GED err", sys.exc_info()[0]

    return distances

def main():
    C = "SELECT ?v WHERE { <http://dbpedia.org/resource/Morocco> <http://dbpedia.org/property/gdpNominalYear> ?v . } LIMIT 1"
    D = ["SELECT ?v WHERE {  <http://dbpedia.org/resource/Connecticut> <http://dbpedia.org/property/governor> ?v . } LIMIT 1", "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT * WHERE { ?city a <http://dbpedia.org/ontology/Place>; rdfs:label 'Madrid'@en.  ?airport a <http://dbpedia.org/ontology/Airport>. {?airport <http://dbpedia.org/ontology/city> ?city} UNION {?airport <http://dbpedia.org/ontology/location> ?city} UNION {?airport <http://dbpedia.org/property/cityServed> ?city.} UNION {?airport <http://dbpedia.org/ontology/city> ?city. }{?airport <http://dbpedia.org/property/iata> ?iata.} UNION  {?airport <http://dbpedia.org/ontology/iataLocationIdentifier> ?iata. } OPTIONAL { ?airport foaf:homepage ?airport_home. } OPTIONAL { ?airport rdfs:label ?name. } OPTIONAL { ?airport <http://dbpedia.org/property/nativename> ?airport_name.} FILTER ( !bound(?name) || langMatches( lang(?name), 'en') )}"]
    print get_distances(C, D)

if __name__ == '__main__':
    main()
