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

def get_result_size(response):
	try:
		respJson = response.json()
		result_size = len(respJson['results']['bindings'])
	except:
		respJson = fallback_json
		result_size = 0
	return result_size


def run_log(query_line):
	# open queries and regex for links
	url_ = re.findall('"GET (.*?) HTTP', query_line)

	if len(url_) == 1:
		request_url = url_[0]

		query_times = []
		resp = ''
		result_size = 0
		
		try:
			for _ in range(11):
				response, exec_time = run_http_request(request_url)
				query_times.append(exec_time)
				
				result_size = get_result_size(response)
				if result_size == 0:
					break
				
				time.sleep(random.random()*2)
		except:
			exec_time = -1

		if exec_time != -1 and result_size > 0:
			cold_exec_time = query_times[0]
			warm_times = query_times[1:]
			warm_mean = np.mean(warm_times, dtype=np.float64)

			query_clean = cleanup_query(request_url)
			return (query_clean + '\t' + str(warm_mean) + '\t' + str(cold_exec_time) + '\t' + str(result_size) + '\n')
	
def worker_pool(log_file):
	results = []
	with open(log_file) as f:

		pool = Pool()
		results = pool.map_async(run_log, f,1 )
		pool.close()
		while not results.ready():
			remaining = results._number_left
			print "Waiting for", remaining, "tasks to complete..."
			sys.stdout.flush()
			time.sleep(10)

	with open(log_file + '-out', 'a') as out:
		for entry in results.get():
			if entry is not None:
				out.write(str(entry))


def main():
	worker_pool('log160k.log')


if __name__ == '__main__':
	main()
