package com.rc;

import java.util.Random;

public class Main {

	static Random rng = new Random() ;
	public static void main(String[] args) {
		try {
			HopfieldNetwork net = new HopfieldNetwork( 35 ) ;
//			net.hebbian(patterns);
			net.storkey(patterns);
			for( double pattern[] : patterns ) {
				test(net, pattern ) ;
			}
		} catch( Throwable t ) {
			t.printStackTrace( );
			System.exit(1); 
		}
	}

	static double [] prepare( double in[] ) {
		double rc[] = new double[ in.length ] ;
		for( int i=0 ; i<rc.length ; i++ ) {
			rc[i] = in[i] == 0 ? -1 : 1 ;
		}
		return rc ;
	}

	static void test( HopfieldNetwork net, double pattern[] ) {

		StringBuilder sb[] = new StringBuilder[7] ;
		
		for( int i=0 ; i<pattern.length ; i++ ) {
			if( rng.nextInt(5) == 0 ) {
				pattern[i] = -pattern[i] ;
			}
		}

		int ix = 0 ;
		for( int i=0 ; i<7 ; i++ ) {
			sb[i] = new StringBuilder() ;
			for( int j=0 ; j<5 ; j++ ) {
				sb[i].append( pattern[ix++] > 0 ? '*' : ' ' );
			}
			sb[i].append( "       " ) ;
		}

		long start = System.currentTimeMillis() ;
		net.run( pattern, 20 ) ;
		long delta = System.currentTimeMillis() - start ;
		//System.out.println( "Run in " + delta + "mS" );
			
		ix = 0 ;
		for( int i=0 ; i<7 ; i++ ) {
			for( int j=0 ; j<5 ; j++ ) {
				sb[i].append( pattern[ix++] > 0 ? '*' : ' ' );
			}
		}
		for( int i=0 ; i<7 ; i++ ) {
			System.out.println( sb[i].toString() ) ;
		}
		System.out.println( "----------------" ) ;
	}

	static double pattern0[] = {
			 0,  0,  1,  0,  0,
			 0,  1,  0,  1,  0,
			 1,  0,  0,  0,  1,
			 1,  1,  1,  1,  1,
			 1,  0,  0,  0,  1,
			 1,  0,  0,  0,  1,
			 1,  0,  0,  0,  1 } ;

	static double pattern1[] = {
			 1,  0,  0,  0,  1,
			 1,  0,  0,  0,  1,
			 1,  0,  0,  0,  1,
			 1,  0,  0,  0,  1,
			 1,  0,  0,  0,  1,
			 1,  0,  0,  0,  1,
			 0,  1,  1,  1,  0 } ;

	static double pattern2[] = {
			 1,  0,  0,  0,  1,
			 1,  0,  0,  0,  1,
			 0,  1,  0,  1,  0,
			 0,  0,  1,  0,  0,
			 0,  1,  0,  1,  0,
			 1,  0,  0,  0,  1,
			 1,  0,  0,  0,  1 } ;

	static double pattern3[] = {
			 0,  1,  1,  1,  0,
			 1,  0,  0,  0,  1,
			 0,  1,  0,  0,  0,
			 0,  0,  1,  0,  0,
			 0,  0,  0,  1,  0,
			 1,  0,  0,  0,  1,
			 0,  1,  1,  1,  0 } ;


	static double patterns[][] = {  prepare(pattern0), 
									prepare(pattern1), 
									prepare(pattern2), 
									prepare(pattern3) } ;

}
