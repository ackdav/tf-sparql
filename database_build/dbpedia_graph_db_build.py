from __future__ import absolute_import
import re, sys, requests, time
from multiprocessing.dummy import Pool 

from sparql_graph.query_vector_converter import convert_query

def write_db(query, time, result_size):
	query_vec = convert_query(query)

	if query_vec != -1:
		query_vec.insert(len(query_vec), time)
		# write db

		# print query_freq
		# sys.stdout.flush()
		return(query + '\t' + str(query_vec) + '\t' + str(time) + '\t' + str(result_size) + '\n')

def preprocess_db_call(line):
	strip = line.strip('\n')
	strip = strip.replace('"', '\\"')
	split = strip.split('\t')
	query = split[0]
	time = split[1]
	result_size = split[2]
	result = write_db(query, time, result_size)
	return result

def run_log(log_file):
	# open queries and regex for links
	results = []
	with open(log_file) as f:
		pool = Pool()
		results = pool.map(preprocess_db_call, f, 1)

	with open('tf-db-cold-1k.txt', 'a') as out:
		for entry in results:
			out.write(entry)

def main():
	run_log('db-cold-novec-1k.txt')

if __name__ == '__main__':
	main()