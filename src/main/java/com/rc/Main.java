package com.rc;

import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
	final static Logger log = LoggerFactory.getLogger( LinearRegression.class ) ;

	static Random rng = new Random() ;
	static final int THETAS_TO_SHOW = 5 ;
	public static void main(String[] args) {
		try {
			//			HopfieldNetwork net = new HopfieldNetwork( 35 ) ;
			//			net.learn(patterns);
			//			test( net, patterns ) ;

			LinearRegression lr = new LinearRegression() ;

			testLR(housePriceX, housePriceY, lr ) ;
			testLR2(housePriceX, housePriceY, lr ) ;

			int ROWS = 500 ;
			int COLS = 100 ;
			double t[] = new double[COLS+1] ;
			double xx[] = new double[ROWS * COLS] ;
			double yy[] = new double[ ROWS ];

			for( int i=0 ; i<t.length ; i++ ) {
				t[i] = rng.nextInt(11) - 6 ;
			}

			//int ix = 0 ;
			for( int j=0 ; j<ROWS ; j++ ) {
				yy[j] = t[0] ;
				for( int i=1 ; i<COLS+1 ; i++ ) {
					int ix = (i-1)*ROWS + j ;
//					log.info( "Next index {}", ix ) ;
					xx[ix] = rng.nextDouble() * 10 ;
					// xx[ix] = (i*100) + j+1 ;
					yy[j] +=  t[i] * xx[ix] ;
					ix++ ;
				}
				yy[j] += rng.nextGaussian() ; 
			}
			// log.info( "Raw data {}", xx ) ;
			// y = f(x)
			// DoubleMatrix m = new DoubleMatrix( ROWS, COLS, xx ) ;
			// System.out.println( m );

			System.out.println( "\n\nActual  >>" + new DoubleMatrix(t).getRows( new IntervalRange(0, Math.min(THETAS_TO_SHOW,t.length) ) ) ); 
			//testLR(ROWS,COLS,xx, yy, lr ) ;
			testLR2(ROWS,COLS,xx, yy, lr ) ;

		} catch( Throwable t ) {
			t.printStackTrace( );
			System.exit(1); 
		}
	}


	static void testLR( double x[][], double y[], LinearRegression lr ) {
		DoubleMatrix I = DoubleMatrix.ones( x.length ) ;
		DoubleMatrix X = DoubleMatrix.concatHorizontally(I, new DoubleMatrix(x)) ;
		testLR( X, new DoubleMatrix(y), lr ) ;
	}
	static void testLR( int ROWS, int COLS, double x[], double y[], LinearRegression lr ) {
		DoubleMatrix I = DoubleMatrix.ones( ROWS ) ;
		DoubleMatrix X = DoubleMatrix.concatHorizontally(I, new DoubleMatrix( ROWS, COLS, x)) ;
		testLR( X, new DoubleMatrix(y), lr ) ;
	}

	static void testLR( DoubleMatrix X, DoubleMatrix y, LinearRegression lr ) {
		System.out.println( " ------------ Nelder Meads -------------" ) ; 
		double scale = 1.0 ;
		log.info( "Scale {}", scale ) ; 
		long start = System.nanoTime() ;
		DoubleMatrix theta = lr.solve( 	
				X, 
				y,
				scale, //1.5,				// Alpha = step rate
				1e-12,			// threshold
				1e-4, 			// L2 lambda
				1000 ) ;		// max Iteration			
		long delta = System.nanoTime() - start ;

		log.debug( "Theta {} x {}", theta.rows, theta.columns ) ;
		log.debug( "    X {} x {}", X.rows, X.columns ) ;

		log.info( "Solved:   {}", theta.getRows( new IntervalRange(0, Math.min(THETAS_TO_SHOW,X.columns) ) ) ) ; 
		log.info( "Solution: {}", X.mmul( theta ).getRows( new IntervalRange(0, Math.min(THETAS_TO_SHOW,X.rows) ) ) ) ; 
		log.info( "Actual:   {}", y.getRows( new IntervalRange(0, Math.min(THETAS_TO_SHOW,X.rows) ) ) ) ;  
		log.info( "Elapsed:  {} uS", delta/1000  );
	}


	static void testLR2( double x[][], double yy[], LinearRegression lr ) {
		DoubleMatrix X = new DoubleMatrix(x) ;
		DoubleMatrix y = new DoubleMatrix(yy) ;
		testLR2( X, y, lr ) ;
	}		

	static void testLR2( int ROWS, int COLS, double x[], double yy[], LinearRegression lr ) {
		DoubleMatrix X = new DoubleMatrix(ROWS, COLS, x) ;
		DoubleMatrix y = new DoubleMatrix(yy) ;
		testLR2( X, y, lr ) ;
	}		

	static void testLR2( DoubleMatrix X, DoubleMatrix y, LinearRegression lr ) {

		System.out.println( " ------------ Least Squares -------------" ) ; 
		long start = System.nanoTime() ;
		DoubleMatrix theta = lr.solveFast( X, y ) ;				
		long delta = System.nanoTime() - start ;

		DoubleMatrix I = DoubleMatrix.ones( X.rows ) ;
		X = DoubleMatrix.concatHorizontally(I, X ) ;

		System.out.println( "Solved:   " + theta.getRows( new IntervalRange(0, Math.min(THETAS_TO_SHOW,X.columns) ) ) ) ; 
		System.out.println( "Solution: " + X.mmul( theta ).getRows( new IntervalRange(0, Math.min(THETAS_TO_SHOW,X.rows) ) ) ) ; 
		System.out.println( "Actual:   " + y.getRows( new IntervalRange(0, Math.min(THETAS_TO_SHOW,X.rows) ) ) ) ; 
		System.out.println( "Elapsed:  " + delta/1000 + "uS" );
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
