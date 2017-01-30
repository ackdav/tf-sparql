# Predicting SPARQL Query Performance with TensorFlow

## TensorFlow installation on Debian SLURM cluster without sudo
The official installation guide to install TensorFlow doesn't provide an option without having sudo access. You can either use pip, virtualenv or docker, which all require sudo access to install in the first place. The trick is to install pip on a user level and then install TensorFlow.

`export PATH=/home/user/$USERNAME/.local/bin/:$PATH`  
`wget https://bootstrap.pypa.io/get-pip.py`

(get-pip.py is simply an entire copy of pip)

`python get-pip.py --user`

`.local/bin/pip install --upgrade --user https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl`

This installs TensorFlow on a user level. Please check for a newer version when installing due to breaking changes in upgrades.

## Setup local DBpedia3.9 mirror on Virtuoso

These steps follow mostly [Tutorial1](https://joernhees.de/blog/2015/11/23/setting-up-a-linked-data-mirror-from-rdf-dumps-dbpedia-2015-04-freebase-wikidata-linkedgeodata-with-virtuoso-7-2-1-and-docker-optional/#comment-6451) and the official [Bulkload example](https://virtuoso.openlinksw.com/dataspace/doc/dav/wiki/Main/VirtBulkRDFLoaderExampleDbpedia), but with some extras, since I didn't have sudo access on the slurm cluster.

### Download the data
`mkdir -p /home/user/$USERNAME/data/datasets/dbpedia3.9
cd /home/user/$USERNAME/data/datasets/dbpedia3.9`

`wget -r -nc -nH --cut-dirs=1 -np -l1 -A '*.nt.bz2' -A '*.owl' -R '*unredirected*' http://downloads.dbpedia.org/3.9/{en/,dbpedia_3.9.owl}`

since virtuoso only accepts `*.bz2`files we need to repackage the files.

switch to an interactive session

`for d in * ; do for i in "${d%/}"/*.bz2 ; do bzcat "$i" | gzip > "${i%.bz2}.gz" && rm "$i" ; done & done`

cd into and make sure `*.bz2` are all removed.

#### Bulkload
Open an interactive session, start Virtuoso (in the background):
`/home/user/$USERNAME/env/bin/virtuoso-t -c "/home/user/$USERNAME/env/var/lib/virtuoso/db/virtuoso.ini"`

Save the `rdfloader.sql`-script from [here](https://virtuoso.openlinksw.com/dataspace/doc/dav/wiki/Main/VirtBulkRDFLoaderScript).  

Start a iSQL session:
`/home/user/ackdav/env/bin/isql-v`

Load the rdfloader script:  
`LOAD rdfloader.sql;`

Add your data for the rdfloader to load:  
`ld_add ('/home/user/$USERNAME/data/datasets/dbpedia3.9/dbpedia_3.9.owl', 'http://dbpedia.org/resource/classes#');`  
`ld_dir_all ('/home/user/$USERNAME/data/datasets/dbpedia3.9/en', '*.*', 'http://dbpedia.org');`

Double check if all files were added:  
`select * from DB.DBA.LOAD_LIST;`  
(-- if unsatisfied use:
-- `delete from DB.DBA.LOAD_LIST;`)

Next step would be to run `rdf_loader_run();`, but this takes over 1h and we don't want to hit the timelimit on the interactive session.

So setup a batchscript with a good 2-3hours of a node. Create a txt file with `rdf_loader_run();` and insert in the batchscript:

`/home/user/ackdav/env/bin/isql-v 127.0.0.1:1111 dba dba /path/to/rdfload.txt`

After it's finished open virtuoso on a node and run either in iSQL or on http://node01.ifi.uzh.ch/sparql:
`sparql SELECT count(*) WHERE { ?s ?p ?o } ;`, where you should see a new dbpedia graph.
