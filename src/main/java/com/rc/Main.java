package com.rc;

import java.util.Random;
import joptsimple.OptionParser;
import joptsimple.OptionSet;

public class Main {

	static Random rng = new Random() ;
	public static void main(String[] args) {
		OptionParser parser = new OptionParser( "hs" ) ;
		OptionSet options = parser.parse( args ) ;
							
		try {
			HopfieldNetwork net = new HopfieldNetwork( 35 ) ;
			if( options.has( "h" ) ) {
				net.hebbian(patterns) ;
			} else if( options.has( "s" ) ) {
				net.storkey(patterns);
			} else {
				System.err.println( "Use -h | -s  for learning (hebbian | storkey)" ) ;
				System.exit( 2 ) ;
			}
			test( net, patterns ) ;
		} catch( Throwable t ) {
			t.printStackTrace( );
			System.exit(1); 
		}
	}


	static void test( HopfieldNetwork net, double patterns[][] ) {

		StringBuilder sb[][] = new StringBuilder[2][7] ;
		for( int i=0 ; i<7 ; i++ ) {
			sb[0][i] = new StringBuilder() ;
			sb[1][i] = new StringBuilder() ;
		}	

		for( double pattern[] : patterns ) {
			for( int i=0 ; i<pattern.length ; i++ ) {
				if( rng.nextInt(3) == 0 ) {
					pattern[i] = rng.nextGaussian() ; //-pattern[i] ;
				}
			}

			int ix = 0 ;
			for( int i=0 ; i<7 ; i++ ) {
				for( int j=0 ; j<5 ; j++ ) { 
					double c = pattern[ix++] ;
					if( c < -0.95 ) sb[0][i].append( ' ' );
					else if( c< -0.50 ) sb[0][i].append( '-' );
					else if( c< -0.0 ) sb[0][i].append( '.' );
					else if( c< 0.5 ) sb[0][i].append( ',' );
					else sb[0][i].append( '*' );
				}
				sb[0][i].append( "       " ) ;
			}

			long start = System.currentTimeMillis() ;
			net.run( pattern, 40 ) ;
			long delta = System.currentTimeMillis() - start ;
			//System.out.println( "Run in " + delta + "mS" );
				
			ix = 0 ;
			for( int i=0 ; i<7 ; i++ ) {
				for( int j=0 ; j<5 ; j++ ) {
					sb[1][i].append( pattern[ix++] > 0 ? '*' : ' ' );
				}
				sb[1][i].append( "       " ) ;
			}
		}
		for( int i=0 ; i<7 ; i++ ) {
			System.out.println( sb[0][i].toString() ) ;
		}
		System.out.println( "----------------" ) ;
		for( int i=0 ; i<7 ; i++ ) {
			System.out.println( sb[1][i].toString() ) ;
		}
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


	static double [] prepare( double in[] ) {
		double rc[] = new double[ in.length ] ;
		for( int i=0 ; i<rc.length ; i++ ) {
			rc[i] = in[i] == 0 ? -1 : 1 ;
		}
		return rc ;
	}
								
}
