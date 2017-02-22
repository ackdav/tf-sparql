import sys
import re
import glob

def readout_feature():
	result = [];

	with open("union-queries.txt", 'a') as f:
		queries = []
		#filter queries out of one log example

		# wildcards to match every extracted file
		for file in glob.glob('.dataset/query-bioportal.log-*-sparql'):
			queries += re.findall("##### (.*?)#####", open(file).read(), re.DOTALL)

		# queries = queries1 + queries2 + queries3 + queries4
		#filter out number of variables in select statement
		for index, query in enumerate(queries):
			
			# regex exec time and substring with select variables
			select_where = re.findall('SELECT(.*?)WHERE', query, re.DOTALL)		
			exec_time = re.findall('p[0-9]{4,6}: (.*?)s, ', query)
			for index, query in enumerate(select_where):
				
				select_variables = query.count("?")

				# some inconsistencies with dataset causes some rare 2 exec times 
				if select_variables>0 and len(exec_time)==1:

					# escape from list
					exec_time = float(exec_time[0])
					f.write(str(select_variables) + '\t' + str(exec_time) + '\n')

					result.append(str(select_variables) + '\t' + str(exec_time) + '\n')
	return result

def main():
	res = readout_feature()

if __name__ == '__main__':
	main()