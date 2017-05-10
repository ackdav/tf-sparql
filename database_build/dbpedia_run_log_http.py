from multiprocessing import Pool
import re, sys, requests, random, json
import time as timesleep
import numpy as np
from tqdm import *
from urlparse import urlparse, parse_qs
import urllib
from datetime import datetime, time

fallback_json = { "head": { "link": [], "vars": ["property", "propertyLabel", "propertyVal", "propertyValLabel"] },
  "results": { "distinct": False, "ordered": True, "bindings": [ ] } }

def run_http_request(req):
	'''Executes HTTP request to server and returns time

	Keyword-args:
	req -- sparql query in url formatting
	'''
	url = 'http://claudio11.ifi.uzh.ch:8890' + req + '&format=json'
	t0 = datetime.utcnow()

	# make call and measure time taken
	resp = requests.get(url)
	time1 = (datetime.utcnow() - t0).total_seconds()

	return resp, time1

def cleanup_query(query):
	'''Cleans log-url into readable sparql query

	Keyword-args:
	query -- log-url to clean
	'''
	line_no_tabs = re.sub(r'%09|%0B', '+', query)
	line_single_spaces = re.sub(r'\++', '+', line_no_tabs)
	line_no_formatting = re.sub(r'%0A|%0D', '', line_single_spaces)
	line_noprefix = re.sub(r'.*query=', '', line_no_formatting)
	line_noquotes = re.sub(r'"', '', line_noprefix)
	line_end_format = re.sub(r'(&.*?)$', '', line_noquotes)

	return urllib.unquote_plus(line_end_format.encode('ascii'))

def get_result_size(response):
	try:
		result_size = len(response['results']['bindings'])
	except:
		# respJson = fallback_json
		result_size = 0
	return result_size

def run_log(query_line, last_timestamp):
	# open queries and regex for links
	url_ = re.findall('"GET (.*?) HTTP', query_line)

	last_timestamp_new = datetime.utcnow()
	if len(url_) == 1:
		request_url = url_[0]
		query_times = []
		resp = ''
		result_size = 0
		
		try:
			utcnow = datetime.utcnow()
			midnight_utc = datetime.combine(utcnow.date(), time(0))

			delta_last_query = (datetime.utcnow() - last_timestamp).total_seconds()

			for _ in range(11):
				response, exec_time = run_http_request(request_url)
				# if exec_time == -1.:
				# 	break

				query_times.append(exec_time)
				# timesleep.sleep(random.random()*0.1)

			last_timestamp_new = datetime.utcnow()
			timestamp_query = ((last_timestamp_new - midnight_utc).total_seconds())

			respJson = response.json()
			result_size = get_result_size(respJson)
		
		except:
			exec_time = -1

		if exec_time != -1 and len(query_times) == 11: #and result_size > 0:
			cold_exec_time = query_times[0]
			warm_times = query_times[1:]
			warm_mean = np.mean(warm_times, dtype=np.float64)

			time_vec = [timestamp_query, delta_last_query]

			query_clean = cleanup_query(request_url)

			res = str(query_clean + '\t'+ str(time_vec) + '\t' + str(warm_mean) + '\t' + str(cold_exec_time) + '\t' + str(result_size) + '\n')
			return (res, last_timestamp_new)

		else:
			return (-1., last_timestamp_new)
	else:
		return (-1., last_timestamp_new)

def main():
	results = []
	log_file = 'database.log'
	# with open(log_file) as f:
	# 	#Spawn pool of workers to execute http queries
	# 	pool = Pool()
	# 	results = pool.map_async(run_log, f,1)
	# 	pool.close()
	# 	while not results.ready():
	# 		remaining = results._number_left
	# 		print "Waiting for", remaining, "tasks to complete..."
	# 		sys.stdout.flush()
	# 		time.sleep(10)

	with open(log_file) as in_, tqdm(total=40000) as pbar:
		count = 0.
		last_timestamp = datetime.utcnow()
		for l_ in in_:
			count += 1
			res, last_timestamp = run_log(l_, last_timestamp)

			if len(results) > 40000:
				break
			if count == 19:
				count = 0
				pbar.update(19)
				sys.stdout.flush()
			if res != -1.:
				results.append(res)

	with open(log_file + '-test2', 'a') as out:
		for entry in results:
		# for entry in results.get():
			if entry is not None:
				out.write(str(entry))

if __name__ == '__main__':
	main()
