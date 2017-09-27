package com.rc;

import java.util.Random;

public class Main {

	static Random rng = new Random() ;
	public static void main(String[] args) {
		try {
			HopfieldNetwork net = new HopfieldNetwork( 35 ) ;
			net.hebbian(patterns);
			test(net, a_pattern ) ;
			test(net, u_pattern ) ;
			test(net, t_pattern ) ;
			test(net, s_pattern ) ;
		} catch( Throwable t ) {
			t.printStackTrace( );
			System.exit(1); 
		}
	}


	static void test( HopfieldNetwork net, double inPattern[] ) {
			
		double pattern[] = new double[35] ;
		for( int i=0 ; i<pattern.length ; i++ ) {
			pattern[i] = (rng.nextInt(6) == 0 ? -1 : 1 ) * inPattern[i] ;
		}

		StringBuilder sb[] = new StringBuilder[7] ;
			
		int ix = 0 ;
		for( int i=0 ; i<7 ; i++ ) {
			sb[i] = new StringBuilder() ;
			for( int j=0 ; j<5 ; j++ ) {
				sb[i].append( pattern[ix++] > 0 ? '*' : ' ' );
			}
			sb[i].append( "       " ) ;
		}

		long start = System.currentTimeMillis() ;
		net.run( pattern, 30 ) ;
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

	static double a_pattern[] = {
			-1, -1, 1, -1, -1,
			-1, 1, -1, 1, -1,
			1, -1, -1, -1, 1,
			1, 1, 1, 1, 1,
			1, -1, -1, -1, 1,
			1, -1, -1, -1, 1,
			1, -1, -1, -1, 1 } ;

	static double u_pattern[] = {
			1, -1, -1, -1, 1,
			1, -1, -1, -1, 1,
			1, -1, -1, -1, 1,
			1, -1, -1, -1, 1,
			1, -1, -1, -1, 1,
			1, -1, -1, -1, 1,
			-1, 1, 1, 1, -1 } ;

	static double t_pattern[] = {
			1, 1, 1, 1, 1,
			-1, -1, 1, -1, -1,
			-1, -1, 1, -1, -1,
			-1, -1, 1, -1, -1,
			-1, -1, 1, -1, -1,
			-1, -1, 1, -1, -1,
			-1, -1, 1, -1, -1 } ;

	static double s_pattern[] = {
			-1, 1, 1, 1, -1,
			1, -1, -1, -1, 1,
			-1, 1, -1, -1, -1,
			-1, -1, 1, -1, -1,
			-1, -1, -1, 1, -1,
			1, -1, -1, -1, 1,
			-1, 1, 1, 1, -1 } ;


	static double patterns[][] = { a_pattern, u_pattern, t_pattern, s_pattern } ;

}
