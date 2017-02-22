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

# print parse_sexp("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT * WHERE { {?city rdfs:label 'Czech Republic'@en.} UNION { ?alias <http://dbpedia.org/property/redirect> ?city; rdfs:label 'Czech Republic'@en. } UNION { ?alias <http://dbpedia.org/property/disambiguates> ?city; rdfs:label 'Czech Republic'@en. } OPTIONAL { ?city <http://dbpedia.org/ontology/abstract> ?abstract} OPTIONAL { ?city geo:lat ?latitude; geo:long ?longitude%7 DOPTIONAL { ?city foaf:depiction ?image } OPTIONAL { ?city rdfs:label ?name } OPTIONAL { ?city foaf:homepage ?home } OPTIONAL { ?city <http://dbpedia.org/ontology/populationTotal> ?population } OPTIONAL { ?city <http://dbpedia.org/ontology/thumbnail> ?thumbnail } FILTER (langMatches( lang(?abstract), 'en'))}")