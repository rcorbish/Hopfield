package com.rc;

import java.util.Random;

import org.jblas.DoubleMatrix;

public class Main {

	static Random rng = new Random() ;
	public static void main(String[] args) {
		try {
//			HopfieldNetwork net = new HopfieldNetwork( 35 ) ;
//			net.learn(patterns);
//			test( net, patterns ) ;
			
			LinearRegression lr = new LinearRegression() ;
			double X[][] = { {2}, {3}, {6}, {1}, {4} } ;
			double y[] = {  5, 7, 13, 3, 9 } ;
			// y = 2x
			
			DoubleMatrix theta = lr.solve( 	new DoubleMatrix(housePriceX), 
						new DoubleMatrix(housePriceY),
						0.0000007,   // learning rate
						0.005, 	// L2 lambda
						0.001,	// threshold
						400 ) ;	// max Iteration
			
			System.out.println( "Solve " + theta );
			
			theta = lr.solveFast( 
					new DoubleMatrix(housePriceX), 
					new DoubleMatrix(housePriceY) 
					) ;
			
			System.out.println( "FSolve " + theta );
			
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

			long start = System.nanoTime() ;
			net.run( pattern, 20 ) ;
			long delta = System.nanoTime() - start ;
			System.out.println( "Run in " + (delta/1000) + "uS" );
				
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

	static double housePriceX[][] = {  
			{2104,5,1,45}, 
			{1416,3,2,40}, 
			{1534,3,2,30}, 
			{ 852,2,1,36} 
			} ; 
	static double housePriceY[] = { 460, 232, 315, 178 } ; 
}
