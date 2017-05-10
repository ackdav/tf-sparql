import sys, re, ast
from collections import deque
query_data = []
with open('random200k.log-result') as in_:
	queue = deque([])
	for id, line in enumerate(in_):
		query_line = line.strip('\n')
		query_line = query_line.split('\t')
		query_vec = ast.literal_eval(query_line[1])

		queue.append(int(query_line[4]))
		if id > 4:
			queue.popleft()
			del query_vec[-1]
			que = list(queue)
			query_vec = query_vec + que

			query_data.append(str(query_line[0]) + '\t' + str(query_vec) + '\t' + str(query_line[2]) + '\t' + str(query_line[3]) + '\n')

with open('random200k.log-rnn', 'a') as out_:
	for l_ in query_data:
		out_.write(l_)