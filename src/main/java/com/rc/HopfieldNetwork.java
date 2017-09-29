package com.rc;

import java.util.Random;

public class HopfieldNetwork {

	final int vectorSize ;
	final double weights[][] ;
	final Random rng ;

	public HopfieldNetwork( final int vectorSize ) {
		this.rng = new Random( 128  ) ;
		this.vectorSize = vectorSize ;
		this.weights = new double[vectorSize][vectorSize] ;
	}

	public void run( double input[], int iterations ) {
		for( int i=0 ; i<iterations ; i++ ) {
			step( input ) ;
		}
	}

	public void step( double input[] ) {

		// shuffle ... part 1
		int indices[] = new int[vectorSize] ;
		for( int i=0 ; i<indices.length ; i++ ) {
			indices[i] = i ;
		}

		for( int i=0 ; i<vectorSize ; i++ ) {
			// shuffle ... part 2
			int ix = rng.nextInt( vectorSize-i ) ;
			indices[ix] = indices[ vectorSize-i-1 ] ;

			input[ix] = getOutput(ix, input ) ;			
		}
	}

	public double getOutput( int neuronIndex, double pattern[] ) {
		double s = 0 ;
		for( int i=0 ; i<vectorSize ; i++ ) {
			s += weights[neuronIndex][i] * pattern[i] ;  
		}
		return s>0 ? 1 : -1  ;
	}

//=====================================================================

	public double[] calculateWeights( int neuronIndex, double patterns[][] ) {

		double weights[] = new double[ vectorSize ] ;

		for( int i=0 ; i<weights.length ; i++ ) {
			if( neuronIndex != i ) {
				double tot = 0.0 ;
				for( int p=0 ; p<patterns.length ; p++ ) {
					tot += patterns[p][i] * patterns[p][neuronIndex] ;
				}
				weights[i] = tot / patterns.length ;
			}
		}

		return weights ;
	}



	public void hebbian( double patterns[][] ) {

		double tmp[][] = new double[vectorSize][] ;

		for( int i=0 ; i<vectorSize ; i++ ) {
			tmp[i] = calculateWeights(i, patterns) ;
		}

		for( int i=0 ; i<this.weights.length ; i++ ) {
			for( int j=0 ; j<this.weights[i].length ; j++ ) {
				this.weights[i][j] = tmp[i][j] ;
			}
		}
	}


	//=================================================================================


	public void storkey( double pattern[] ) { 

		double h[] = new double[vectorSize] ;
		double dw[][] = new double[vectorSize][vectorSize] ;
		
		// h = pattern x weights 
		for(int i=0;i<vectorSize;i++) {
			for(int k=0;k<vectorSize;k++) {
				if(k!=i) {
					h[i] += weights[i][k] * pattern[k];
				}
			}
		}

		// t1 = pattern x pattern
		// t2 = pattern x h
		// t3 = n/a
		for(int i=0;i<vectorSize;i++) {
			for(int j=0;j<vectorSize;j++) {
				double t1 = pattern[i] * pattern[j] ;
				double t2 = pattern[i] * h[j] ;
				double t3 = pattern[j] * h[i] ;
				dw[i][j] += (t1-t2-t3) / vectorSize ;
			}
		}		

		for( int i=0 ; i<vectorSize ; i++ ) {
			for( int j=0 ; j<vectorSize ; j++ ) {
				weights[i][j] += dw[i][j] ;
			}
		}
	}
	

	public void storkey( double patterns[][] ) {
		for( int i=0 ; i<vectorSize ; i++ ) {
			for( int j=0 ; j<vectorSize ; j++ ) {
				weights[i][j] = 0 ;
			}
		}
		for( int p=0 ; p<patterns.length ; p++ ) {
			storkey( patterns[p] ) ;
		}
	}

	
}
