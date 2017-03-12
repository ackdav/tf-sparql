from multiprocessing import Pool
import re, sys, requests, time, random
import numpy as np

from urlparse import urlparse, parse_qs
import urllib

fallback_json = { "head": { "link": [], "vars": ["property", "propertyLabel", "propertyVal", "propertyValLabel"] },
  "results": { "distinct": False, "ordered": True, "bindings": [ ] } }

def run_http_request(req):
	url = 'http://claudio11.ifi.uzh.ch:8890' + req + '&format=json'
	t0 = time.clock()
	# make call and measure time taken
	resp = requests.get(url)
	time1 = time.clock() - t0
	return resp, time1

def cleanup_query(query):
	line_no_tabs = re.sub(r'%09|%0B', '+', query)
	line_single_spaces = re.sub(r'\++', '+', line_no_tabs)
	line_no_formatting = re.sub(r'%0A|%0D', '', line_single_spaces)
	line_noprefix = re.sub(r'.*query=', '', line_no_formatting)
	line_noquotes = re.sub(r'"', '', line_noprefix)
	line_end_format = re.sub(r'(&.*?)$', '', line_noquotes)

	#decode url nicely into SPARQL query
	return urllib.unquote_plus(line_end_format.encode('ascii'))


def write_db(result_list, log_file):
	with open(log_file + '-out', 'a') as out:
		for result in result_list:
			query, time, result_size = result
			# write db
			out.write(decoded + '\t' + '\t' + str(time) + '\t' + str(result_size) + '\n')
			print 'wrote query '

def run_log(query_line):
	# open queries and regex for links
	url_ = re.findall('"GET (.*?) HTTP', query_line)

	if len(url_) == 1:
		request_url = url_[0]

		query_times = []
		resp = ''
		for _ in range(11):
			response, exec_time = run_http_request(request_url)
			query_times.append(exec_time)
			resp = response
			time.sleep(random.random()*2)
		query_times = query_times[1:]
		exec_time = np.mean(query_times, dtype=np.float64)

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
			query_clean = cleanup_query(request_url)
			return (query_clean + '\t' + str(exec_time) + '\t' + str(result_size) + '\n')
	

def worker_pool(log_file):
	results = []
	with open(log_file) as f:

		pool = Pool()
		results = pool.map_async(run_log, f,1 )
		pool.close()
		while not results.ready():
			remaining = results._number_left
			print "Waiting for", remaining, "tasks to complete..."
			time.sleep(10)

	with open(log_file + '-out', 'a') as out:
		for entry in results.get():
			if entry is not None:
				out.write(str(entry))


def main():
	worker_pool('log20.txt')


if __name__ == '__main__':
	main()
