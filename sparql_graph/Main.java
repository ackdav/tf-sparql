import java.io.BufferedReader;
import java.io.FileReader;

import org.apache.jena.shared.PrefixMapping;
import org.apache.jena.sparql.algebra.*;
import org.apache.jena.query.*;

public class Main {

	static String appendPrefixes(String query) {
		String prefix = new String();

		try (BufferedReader br = new BufferedReader(new FileReader("jena-missing-prefixes.txt"))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		       prefix += line;
		    }
		    prefix += ' ';
		}
		catch (Exception e){
			System.out.println("Error: " + e);
		}

		query = prefix + query;
		return query;
	}

    public static void main(String args[]){
    	String appendedQuery = appendPrefixes(args[0]);
	    Query query = QueryFactory.create(appendedQuery);
	    Op op = Algebra.compile(query) ;
        System.out.println(op);
    }
}
