import sys
import re

with open("union-queries.txt", 'a') as f:

	queries = re.findall("^(.*?)\t", open('dbpedia-20k-warm.txt').read())

	print "numba" + str(len(queries))
	print queries

	# for query in range(len(queries)):

	# 	print query

		# if query.count("UNION") > 0:
		# 	print "hoihoi"
print 'done'
