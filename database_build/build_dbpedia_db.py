import re
import sys
import requests
from time import time
from urlparse import urlparse, parse_qs
import urllib

fallback_json = { "head": { "link": [], "vars": ["property", "propertyLabel", "propertyVal", "propertyValLabel"] },
  "results": { "distinct": False, "ordered": True, "bindings": [ ] } }

with open("tf-db.txt", 'a') as f:

	# open queries and regex for links
	urls = re.findall('"GET (.*?) HTTP', open ('access.log-20150821').read())

	for index, line in enumerate(urls):
		url = 'http://claudio11.ifi.uzh.ch:8890' + line + '&format=json'

		# make call and measure time taken
		resp = requests.get(url)
		time = resp.elapsed.total_seconds()

		try:
			respJson = resp.json()
		except:
			respJson = fallback_json
		
		# Remove formatting from log to stay consistent (not every url is formatted)
		line_no_tabs = re.sub(r'%09|%0B', '+', line)
		line_single_spaces = re.sub(r'\++', '+', line_no_tabs)
		line_no_formatting = re.sub(r'%0A|%0D', '', line_single_spaces)

		#decode url nicely into SPARQL query
		decoded = urllib.unquote_plus(line_no_formatting.encode('utf-8'))

		#try to fetch result size
		try:
			result_size = len(respJson['results']['bindings'])
		except:
			result_size = 0

		# write db
		f.write(decoded + '\t' + str(time) + '\t' + str(result_size) + '\n')

		print 'wrote query ' + str(index)

print 'done'