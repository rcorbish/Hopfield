package com.rc;

import java.util.Random;
/**
 * Implements a Hopfiled Network, which supports Storkey 
 * and Hebbian learning
 * 
 * The network capacity depends on learning:
 * 
 * Hebb 	num patterns = 0.138 * vectorSize
 * Storkey	num patterns = vectorSize / sqrt( 2 x log(vectorSize) )
 */
public class HopfieldNetwork {

	/**
	 * The length of the vector pattern to 'remember'
	 */
	private final int vectorSize ;
	/**
	 * Weights of all interconnects 
	 * Usually substripted by i,j  the weight from i to j 
	 * neuron
	 */
	private final double weights[][] ;

	// Used in shuffling
	private final Random rng ;

	/**
	 * Create a new network 
	 * 
	 * @param vectorSize the number of neurons to put into the network
	 */
	public HopfieldNetwork( final int vectorSize ) {
		this.rng = new Random( 128  ) ;
		this.vectorSize = vectorSize ;
		this.weights = new double[vectorSize][vectorSize] ;
	}

	/**
	 * Parse a pattern and update it with one of the learned patterns.
	 * This does not have an early finish, iterations of about 10 
	 * work well in many cases.
	 * 
	 * @param pattern the input pattern - value elements should be -1 to +1
	 * @param iterations the number of iterations to execute before finishing
	 */
	public void run( double pattern[], int iterations ) {
		for( int i=0 ; i<iterations ; i++ ) {
			step( pattern ) ;
		}
	}

	/**
	 * Execute one step of sending an input pattern element by
	 * element, in a random order, into the network.
	 * The input pattern is updated with the learned pattern.
	 * This process should be repeated several times to get
	 * a 'good' output
	 * 
	 * @see #run(double[], int)
	 * @param pattern the input pattern - value elements should be -1 to +1
	 * 
	 */
	protected void step( double pattern[] ) {

		// shuffle ... part 1
		int indices[] = new int[vectorSize] ;
		for( int i=0 ; i<indices.length ; i++ ) {
			indices[i] = i ;
		}

		for( int i=0 ; i<vectorSize ; i++ ) {
			// shuffle ... part 2
			int ix = rng.nextInt( vectorSize-i ) ;
			indices[ix] = indices[ vectorSize-i-1 ] ;

			pattern[ix] = getOutput(ix, pattern ) ;			
		}
	}

	/**
	 * Passes a pattern element into the network,
	 * returning the value of the neuron after summing
	 * all inputs, scaled by their weights.
	 * 
	 * @param neuronIndex which neuron to process
	 * @param pattern the input pattern - value elements should be -1 to +1
	 * @return the output, normalized to -1 or +1 
	 */
	protected double getOutput( int neuronIndex, double pattern[] ) {
		double s = 0 ;
		for( int i=0 ; i<vectorSize ; i++ ) {
			s += weights[neuronIndex][i] * pattern[i] ;  
		}
		return s>0 ? 1 : -1  ;
	}

//=====================================================================

	/**
	 * Train the network with a set of patterns, using
	 * Hebb's batch learning method.
	 * 
	 * @param patterns an array of vector patterns
	 */
	public void hebbian( double patterns[][] ) {

		// calculate weights in a single batch
		for( int i=0 ; i<vectorSize ; i++ ) {
			for( int j=0 ; j<vectorSize ; j++ ) {
				if( j != i ) {
					double tot = 0.0 ;
					for( int p=0 ; p<patterns.length ; p++ ) {
						tot += patterns[p][j] * patterns[p][i] ;
					}
					weights[i][j] = tot / patterns.length ;
				}
			}
		}
	}


	//=================================================================================


	protected void storkey( double pattern[] ) { 

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
	
/**
 * Train the network with a set of patterns, using
 * Storkey's iterative learning method.
 * 
 * @param patterns an array of vector patterns
 */
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
