package com.rc;

import java.util.Random;

import org.jblas.DoubleMatrix;
/**
 * Implements a Hopfield Network, which supports Storkey (2nd order) learning
 * 	 
 * http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.954&rep=rep1&type=pdf
 *
 * The network capacity (#patterns) = vectorSize / sqrt( 2 x ln(vectorSize) )
 * 
 * This will also forget older learned patterns if we keep adding new ones.
 * 
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
	private final DoubleMatrix weights ;

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
		this.weights = new DoubleMatrix(vectorSize,vectorSize) ;
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
		DoubleMatrix p = new DoubleMatrix( pattern ) ;
		for( int i=0 ; i<iterations ; i++ ) {
			step( p ) ;
		}
	}

	/**
	 * Execute one step of sending an input pattern element by
	 * element, in a random order, into the network.
	 * The input pattern is updated with the learned pattern.
	 * This process should be repeated several times to get
	 * a 'good' output
	 * 
	 * NB This is not vectorized (yet) due to the shuffling
	 * 
	 * @see #run(double[], int)
	 * @param pattern the input pattern - value elements should be -1 to +1
	 * 
	 */
	protected void step( DoubleMatrix pattern ) {

		// shuffle ... part 1
		int indices[] = new int[vectorSize] ;
		for( int i=0 ; i<indices.length ; i++ ) {
			indices[i] = i ;
		}

		for( int i=0 ; i<vectorSize ; i++ ) {
			// shuffle ... part 2
			int ix = rng.nextInt( vectorSize-i ) ;
			indices[ix] = indices[ vectorSize-i-1 ] ;

			pattern.put(ix, getOutput(ix, pattern ) ) ;			
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
	protected double getOutput( int neuronIndex, DoubleMatrix pattern ) {
		double dp = weights.getColumn( neuronIndex ).dot( pattern ) ;
		return dp>0 ? 1 : -1  ;
	}




	//=================================================================================

	/**
	 * Run Storkey learning on one pattern
	 * 
	 * @see #storkey(double[][])
	 * 
	 * @param pattern
	 */

	protected void learn( DoubleMatrix pattern ) { 

		DoubleMatrix h = weights.mmul( pattern ) ;   // hebbian component
		
		DoubleMatrix t1 = pattern.mmul( pattern.transpose() ) ;
		DoubleMatrix t2 = pattern.mmul( h.transpose() ) ;
		DoubleMatrix t3 = h.mmul( pattern.transpose() ) ;
		DoubleMatrix t4 = h.mmul( h.transpose() ) ;	
	
		DoubleMatrix dw = t1.sub( t2 ).sub( t3 ).add( t4 ).div( vectorSize ) ;
		
		// clear the diagonal ( so we can't learn identity matrix )
		for( int i=0 ; i<vectorSize ; i++ ) dw.put( i, i, 0 ) ;
		
		weights.addi( dw ) ; 
	}
	
/**
 * Train the network with a set of patterns, using
 * Storkey's iterative learning method.
 * 
 * @param patterns an array of vector patterns
 */
	public void learn( double patterns[][] ) {
		weights.fill(0) ;

		for( int p=0 ; p<patterns.length ; p++ ) {
			learn( new DoubleMatrix( patterns[p] ) ) ;
		}
	}
}
