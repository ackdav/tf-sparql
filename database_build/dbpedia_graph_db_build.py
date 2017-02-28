from __future__ import absolute_import
import re, sys, requests, time, random
from multiprocessing.dummy import Pool 
from functools import partial


from sparql_graph.query_vector_converter import convert_query, jena_graph
from tree_edit_distance import get_distances

ged_samples = []

def ged_sample(log_file, number):
	global ged_samples
	# get random sample of ged-distance benchmark queries
	with open(log_file) as f:
		lines = random.sample(f.readlines(), number)

	for line in lines:
		ged_samples.append(preprocess_write_db(False, line))


def ged_distance(query):
	jena_query = jena_graph('Main', query)
	return get_distances(jena_query, ged_samples)

def write_db(query, time, full_convert):
	query_vec = convert_query(query, full_convert)

	if query_vec != -1:
		print '.',
		# insert time at end, if converting the db and not ged-sample-set
		if len(ged_samples)>0 and full_convert:
			distances = ged_distance(query)
			if len(distances) > 0:
				query_vec = query_vec + ged_distance(query)
		if full_convert:
			query_vec.insert(len(query_vec), time)
		# write db
		# print '.',
		# sys.stdout.flush()
		return query_vec

def preprocess_write_db(full_convert, line):
	strip = line.strip('\n')
	strip = strip.replace('"', '\\"')
	split = strip.split('\t')
	query = split[0]
	time = split[1]
	result_size = split[2]
	query_vec = write_db(query, time, full_convert)
	if full_convert:
		return (query + '\t' + str(query_vec) + '\t' + str(time) + '\t' + str(result_size) + '\n')
	else:
		return query_vec

def run_log(log_file):
	# open queries and regex for links
	results = []
	with open(log_file) as f:
		pool = Pool()
		func_partial = partial(preprocess_write_db, True)
		results = pool.map(func_partial, f, 1)
		# for line in f:
		# 	results.append(preprocess_write_db(True, line))
	with open('tf-db-cold-1k.txt', 'a') as out:
		for entry in results:
			out.write(entry)

def main():
	ged_sample('db-cold-novec-200.txt', 5)
	# print sample
	run_log('db-cold-novec-200.txt')

if __name__ == '__main__':
	main()