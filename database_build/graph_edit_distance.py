import  sys, re
from subprocess import STDOUT,PIPE,Popen

def add_missing_prefixes(query):
    with open('jena-missing-prefixes.txt') as f:
        for prefix in f:
            query = prefix + " " + query
    return query

def rewrite_describe_queries(query):
    uri = re.findall('DESCRIBE <(.*?)>', query)
    query = 'CONSTRUCT  { ?subject ?predicate ?object } WHERE { ?subject ?predicate ?object . FILTER (  ?subject = <'+uri[0]+'> || ?object = <'+uri[0]+'>) }'
    return query

def get_distances(query, query_list):
    distances = []
    query = add_missing_prefixes(query)

    if 'DESCRIBE' in query:
        query = rewrite_describe_queries(query)

    for benchmark_query in query_list:
    	try:
            if 'DESCRIBE' in benchmark_query:
                benchmark_query = rewrite_describe_queries(benchmark_query)

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
    C = "DESCRIBE <http://dbpedia.org/resource/abc>"
    D = ["DESCRIBE <http://dbpedia.org/resource/Canning_Vale,_Western_Australia>"]
    print get_distances(C, D)

if __name__ == '__main__':
    main()
