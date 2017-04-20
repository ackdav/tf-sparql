import re, ast

complicated = []

with open('tf-db-cold.txt') as f:
	for line in f:
		splitted = line.split('\t')
		vec = splitted[1]
		vec = unicode(vec)
		vec = ast.literal_eval(vec)
		count = 0
		for e in vec:
			if e == 0:
				count += 1
		if count < 19:
			complicated.append(line)

with open ('verycomp.txt', 'a') as o:
	for _ in complicated:
		for x in _:
			o.write(x)