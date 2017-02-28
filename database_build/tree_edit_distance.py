import zss, sys

try:
    from editdist import distance as strdist
except ImportError:
    def strdist(a, b):
        if a == b:
            return 0
        else:
            return 1

def weird_dist(A, B):
    return 1*strdist(A, B)

class ZSGraph(object):
    def __repr__(self):
        return ' Node ("%s","%s")' % (self.my_label, self.my_children) +''

    def __str__(self):
        return "An instance of class Test with state:\n label=%s \n children=%s" % (self.my_label, self.my_children)

    def __init__(self, label):
        self.my_label = label
        self.my_children = list()

    @staticmethod
    def get_children(node):
        return node[1:]

    @staticmethod
    def get_label(node):
        return node[0]

def get_distances(graph_one, graph_list):
    distances = []
    for graph_two in graph_list:
    	try:
        	distances.append(zss.simple_distance( graph_one[0], graph_two[0], ZSGraph.get_children, ZSGraph.get_label, weird_dist))
        except:
			print "Tree_Distance err", sys.exc_info()[0]
    return distances

def main():
    C = [['project', ['?type'], ['bgp', ['triple', '<http://dbpedia.org/resource/Cityvibe>', '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', '?type']]]]
    D = [['project', ['?label', '?lat', '?long'], ['leftjoin', ['bgp', ['triple', '<http://dbpedia.org/resource/2005_Primera_Divisi%C3%B3n_de_M%C3%A9xico_Apertura>', '<http://www.w3.org/2000/01/rdf-schema#label>', '?label']], ['bgp', ['triple', '<http://dbpedia.org/resource/2005_Primera_Divisi%C3%B3n_de_M%C3%A9xico_Apertura>', '<http://www.w3.org/2003/01/geo/wgs84_pos#lat>', '?lat'], ['triple', '<http://dbpedia.org/resource/2005_Primera_Divisi%C3%B3n_de_M%C3%A9xico_Apertura>', '<http://www.w3.org/2003/01/geo/wgs84_pos#long>', '?long']]]]]
    print get_distances(C, D)

if __name__ == '__main__':
    main()