from __future__ import absolute_import

import re, sys, requests, time

from urlparse import urlparse, parse_qs
import urllib

from sparql_graph.query_vector_converter import convert_query

fallback_json = { "head": { "link": [], "vars": ["property", "propertyLabel", "propertyVal", "propertyValLabel"] },
  "results": { "distinct": False, "ordered": True, "bindings": [ ] } }

def run_http_request(req):
	url = 'http://claudio11.ifi.uzh.ch:8890' + req + '&format=json'
	t0 = time.clock()
	# make call and measure time taken
	resp = requests.get(url)
	time1 = time.clock() - t0

	return resp, time1

def write_db(query, time, result_size):
	with open("tf-db-cold.txt", 'a') as f:
		# Remove formatting from log to stay consistent (not every url is formatted)
		line_no_tabs = re.sub(r'%09|%0B', '+', query)
		line_single_spaces = re.sub(r'\++', '+', line_no_tabs)
		line_no_formatting = re.sub(r'%0A|%0D', '', line_single_spaces)
		line_noprefix = re.sub(r'.*query=', '', line_no_formatting)
		line_noquotes = re.sub(r'"', '', line_noprefix)
		line_end_format = re.sub(r'(&.*?)$', '', line_noquotes)

		#decode url nicely into SPARQL query
		decoded = urllib.unquote_plus(line_end_format.encode('ascii'))

		query_vec = convert_query(decoded, True)
		if query_vec != -1:
			query_vec.insert(len(query_vec), time)
			# write db
			f.write(decoded + '\t' + str(query_vec) + '\t' + str(time) + '\t' + str(result_size) + '\n')
			print 'wrote query '

def run_log(log_file):
	# open queries and regex for links
	urls = re.findall('"GET (.*?) HTTP', open (log_file).read())

	for index, line in enumerate(urls):
		resp, time = run_http_request(line)

		try:
			respJson = resp.json()
		except:
			respJson = fallback_json

		#try to fetch result size
		try:
			result_size = len(respJson['results']['bindings'])
		except:
			result_size = 0
		
		if result_size > 0:
			write_db(line, time, result_size)

		else:
			print '.',

def main():
	run_log('log100k.log')


if __name__ == '__main__':
	main()