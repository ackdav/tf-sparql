import java.io.BufferedReader;
import java.io.FileReader;

import org.apache.jena.sparql.algebra.*;
import org.apache.jena.query.*;
import org.apache.jena.*;

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
		// System.out.println(query);
		return query;
	}

	public static void main(String args[]){
		String noBackslash = args[0].replaceAll("\\\\", "");

		String appendedQuery = appendPrefixes(noBackslash);

		// parser.parse(QueryParser.escape(appendedQuery))
		Query query = QueryFactory.create(appendedQuery);
		Op op = Algebra.compile(query) ;
		System.out.println(op);
	}
}
