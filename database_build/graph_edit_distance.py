import  sys, re, os.path, time
from subprocess import STDOUT,PIPE,Popen

def add_missing_prefixes(query):
    with open(os.path.dirname(os.path.abspath(__file__))+'/jena-missing-prefixes.txt') as f:
        query = 'PREFIX delimiter: <http://delimiter.com/> ' + query
        for prefix in f:
            query = prefix.strip('\n') + " " + query
        query = 'PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> ' + query
    return query

def remove_temp_prefixes(query):
    return re.sub(r'^(.*? <http://delimiter.com/> )', '', query)

def rewrite_describe_queries(query):
    uri = re.findall('DESCRIBE <(.*?)>', query)
    query = 'CONSTRUCT  { ?subject ?predicate ?object } WHERE { ?subject ?predicate ?object . FILTER (  ?subject = <'+uri[0]+'> || ?object = <'+uri[0]+'>) }'
    return query

def get_distances(query):
    distances = []
    query = add_missing_prefixes(query)

    benchmark_queries = []
    with open(os.path.dirname(os.path.abspath(__file__)) + '/dbpedia_stats/dbpedia_benchmark_queries.txt') as f:
        for line in f:
            benchmark_queries.append(line)

    if 'DESCRIBE' in query:
        query = rewrite_describe_queries(query)

    for benchmark_query in benchmark_queries:
    	try:
            benchmark_query = add_missing_prefixes(benchmark_query)
            cmd = [os.path.join(os.path.dirname(__file__), "../../ged-wrap/scripts/qdistance-beam")+" --std \"%s\" \"%s\" --beam 500" % (query, benchmark_query)]
            # print cmd
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
            stdout, stderr = proc.communicate()
            if not 'help' in stdout:
                distances.append(float(stdout))
        except:
            distances.append(-1.0)
    return distances

def main():
    C = "DESCRIBE <http://dbpedia.org/resource/Drummoyne,_New_South_Wales>"
    # D = ["PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX space: <http://purl.org/net/schemas/space/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX dbpedia-prop: <http://dbpedia.org/property/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>  SELECT DISTINCT ?var FROM <http://dbpedia.org> WHERE { ?var5 dbpedia-owl:thumbnail ?var4 . ?var5 rdf:type dbpedia-owl:Person . ?var5 rdfs:label ?var . ?var5 foaf:page ?var8 . OPTIONAL { ?var5 foaf:homepage ?var10 .} . } LIMIT 1000",
        # "PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX : <http://dbpedia.org/resource/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX dbpedia2: <http://dbpedia.org/property/> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX dbo: <http://dbpedia.org/ontology/> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX dbpedia: <http://dbpedia.org/> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT DISTINCT ?var FROM <http://dbpedia.org> WHERE { { ?var ?var5 ?var6 . ?var6 foaf:name ?var8 . } UNION { ?var9 ?var5 ?var ; foaf:name ?var4 . } } LIMIT 1000",
        # "PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX : <http://dbpedia.org/resource/> PREFIX dbpedia2: <http://dbpedia.org/property/> PREFIX dbpedia: <http://dbpedia.org/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT DISTINCT ?var  FROM <http://dbpedia.org> WHERE { ?var3 rdf:type <http://dbpedia.org/class/yago/Company108058098> . ?var3 dbpedia2:numEmployees ?var  . ?var3 foaf:homepage ?var7 . FILTER ( datatype(?var) = <http://www.w3.org/2001/XMLSchema#int> ) } LIMIT 1000",
        # "PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX : <http://dbpedia.org/resource/> PREFIX dbpedia2: <http://dbpedia.org/property/> PREFIX dbpedia: <http://dbpedia.org/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX umbelBus: <http://umbel.org/umbel/sc/Business> PREFIX umbelCountry: <http://umbel.org/umbel/sc/IndependentCountry> SELECT distinct ?var FROM <http://dbpedia.org> WHERE { ?var0 rdfs:comment ?var1. ?var0 foaf:page ?var OPTIONAL{?var0 skos:subject ?var6} OPTIONAL{?var0 dbpedia2:industry ?var5}OPTIONAL{?var0 dbpedia2:location ?var2}OPTIONAL{?var0 dbpedia2:locationCountry ?var3}OPTIONAL{?var0 dbpedia2:locationCity ?var9; dbpedia2:manufacturer ?var0}OPTIONAL{?var0 dbpedia2:products ?var11; dbpedia2:model ?var0}OPTIONAL{?var0 <http://www.georss.org/georss/point> ?var10}OPTIONAL{?var0 rdf:type ?var7}} LIMIT 1000",
        #  "PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX dc: <http://purl.org/dc/elements/1.1/> PREFIX : <http://dbpedia.org/resource/> PREFIX dbpedia2: <http://dbpedia.org/property/> PREFIX dbpedia: <http://dbpedia.org/> PREFIX skos: <http://www.w3.org/2004/02/skos/core#> SELECT DISTINCT ?var0 ?var1 FROM <http://dbpedia.org> { ?var3 foaf:page ?var7. ?var3 rdf:type <http://dbpedia.org/ontology/SoccerPlayer> . ?var3 dbpedia2:position ?var6 . ?var3 <http://dbpedia.org/property/clubs> ?var8. ?var8 <http://dbpedia.org/ontology/capacity> ?var1 . ?var3 <http://dbpedia.org/ontology/birthPlace> ?var5 . ?var5 ?var4 ?var0. OPTIONAL {?var3 <http://dbpedia.org/ontology/number> ?var35.} Filter (?var4 = <http://dbpedia.org/property/populationEstimate> || ?var4 = <http://dbpedia.org/property/populationCensus> || ?var4 = <http://dbpedia.org/property/statPop> ) . Filter (?var6 = 'Goalkeeper'@en || ?var6 = <http://dbpedia.org/resource/Goalkeeper_%28association_football%29> || ?var6 = <http://dbpedia.org/resource/Goalkeeper_%28football%29>) } LIMIT 1000"]
    print get_distances(C)

if __name__ == '__main__':
    main()
