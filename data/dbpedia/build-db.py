import re
import sys
import requests
import time
from urlparse import urlparse, parse_qs
import urllib

fallback_json = { "head": { "link": [], "vars": ["property", "propertyLabel", "propertyVal", "propertyValLabel"] },
  "results": { "distinct": False, "ordered": True, "bindings": [ ] } }

with open("tf-db-cold.txt", 'a') as f:

	# open queries and regex for links
	urls = re.findall('"GET (.*?) HTTP', open ('log20k.log').read())

	for index, line in enumerate(urls):
		url = 'http://claudio11.ifi.uzh.ch:8890' + line + '&format=json'

		t0 = time.clock()
		# make call and measure time taken
		resp = requests.get(url)
		time1 = time.clock() - t0

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
			# Remove formatting from log to stay consistent (not every url is formatted)
			line_no_tabs = re.sub(r'%09|%0B', '+', line)
			line_single_spaces = re.sub(r'\++', '+', line_no_tabs)
			line_no_formatting = re.sub(r'%0A|%0D', '', line_single_spaces)
			line_noprefix = re.sub(r'.*query=', '', line_no_formatting)
			#decode url nicely into SPARQL query
			decoded = urllib.unquote_plus(line_noprefix.encode('ascii'))

			# write db
			f.write(decoded + '\t' + str(time1) + '\t' + str(result_size) + '\n')

			print 'wrote query ' + str(index)

		else:
			print '.',

print 'done'