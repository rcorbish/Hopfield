package com.rc;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LinearRegression {

	final static Logger log = LoggerFactory.getLogger( LinearRegression.class ) ;

	public DoubleMatrix solveFast( DoubleMatrix x,  DoubleMatrix y ) {
		log.debug( "Starting solveFast" ) ;
		
		if( x.isSquare() && y.length==x.rows ) {
		//	return Solve.solve(x, y) ;
		}

		DoubleMatrix I = DoubleMatrix.ones( x.rows ) ;
		DoubleMatrix X = DoubleMatrix.concatHorizontally(I, x) ;
		return Solve.solveLeastSquares(X, y) ;		
	}

	public DoubleMatrix solve( DoubleMatrix x,  DoubleMatrix y, double alpha, double epsilon, double lambda, int maxIterations ) {
		log.debug( "Starting solve" ) ;
		
		DoubleMatrix I = DoubleMatrix.ones( x.rows ) ;
		DoubleMatrix X = DoubleMatrix.concatHorizontally(I, x) ;

		DoubleMatrix theta = DoubleMatrix.zeros( X.columns ) ;
		
		return nelderMead(X, y, theta, alpha, epsilon, lambda, maxIterations ).transpose();
	}


	public double cost( DoubleMatrix X, DoubleMatrix y, DoubleMatrix theta, double lambda ) {
		DoubleMatrix S = X.mmul( theta ).subi( y ) ;		
		double j1 = S.muli(S).sum() ;

		DoubleMatrix t = theta.mul( theta ).muli( lambda ) ;
		double cost = ( t.sum() - t.get(0) + j1 ) / ( 2 * y.length ) ;

		return cost ;
	}



	public DoubleMatrix nelderMead( DoubleMatrix X, DoubleMatrix y, DoubleMatrix start, double scale, double EPSILON, double LAMBDA, int maxIterations ) {

		final double ALPHA  = 1.5;       /* contraction coefficient */
		final double BETA   = 0.5;       /* contraction coefficient */
		final double GAMMA  = 2.0;       /* expansion coefficient */

		int n = X.columns ;
		//double scale = scale ;
		double f[]   = new double[n+1];

		int itr;	    /* track the number of iterations */

		double pn = scale*(Math.sqrt(n+1)-1+n)/(n*Math.sqrt(2));
		double qn = scale*(Math.sqrt(n+1)-1)/(n*Math.sqrt(2));

		DoubleMatrix thetas = new DoubleMatrix( n+1, n ) ;
		for( int i=0;i<n;i++) {
			thetas.putRow(0, start) ;
		}

		for( int i=1;i<=n;i++) {
			thetas.putRow( i, start.add(qn).put( i-1, start.get(i-1)+pn) ) ;
		}
		
		for( int j=0;j<=n;j++) {
			f[j] = cost( X, y, thetas.getRow(j).transpose(), LAMBDA ) ;
		}

		
		/* begin the main loop of the minimization */
		for( itr=0 ; itr<maxIterations ; itr++ ) {     

			int vs=0 ;	// smallest
			int vg=0 ;	// worst
			int vh=0 ;	// second worst
			
			for( int j=1;j<=n;j++) {
				if (f[j] < f[vs]) {
					vs = j;
				}
				if (f[j] > f[vg]) {
					vg = j;
				}
			}
			for( int j=1; j<=n; j++) {
				if (f[j] > f[vh] && f[j] < f[vg] ) {
					vh = j;
				}
			}			
			
			DoubleMatrix vm = thetas.columnSums().subi( thetas.getRow(vg) ).div( n ) ; 
			
			DoubleMatrix vr = vm.sub( thetas.getRow(vg) ).muli( ALPHA ).addi( vm ).reshape( vm.columns, vm.rows ) ;
			double fr = cost( X, y, vr, LAMBDA ) ;

			if( fr < f[vh] && fr >= f[vs]) {
				thetas.putRow(vg, vr);
				f[vg] = fr ;
			}
			
			/* investigate a step further in this direction */
			if ( fr <  f[vs]) {
				DoubleMatrix ve = vr.sub( vm ).muli( GAMMA ).addi( vm ).reshape( vm.columns, vm.rows ) ;

				double fe = cost( X, y, ve, LAMBDA );
				if (fe < f[vs] ) {
					f[vg] = fe;
					thetas.putRow( vg, ve ) ;
				}
				else {
					f[vg] = fr;
					thetas.putRow( vg, vr ) ;
				}
			}

			
			/* check to see if a contraction is necessary */
			if (fr >= f[vh]) {
				DoubleMatrix vc = (fr < f[vg] && fr >= f[vh])  ?					
					vr.sub( vm ).muli( BETA ).addi( vm ) :   							// outside contraction 					
					vm.sub( thetas.getRow(vg) ).muli( -BETA ).addi( vm ).reshape( vm.columns, vm.rows ) ;  // inside				
				
				double fc = cost( X, y, vc, LAMBDA );

				if (fc < f[vg]) {
					thetas.putRow( vg, vc ) ;
					f[vg] = fc;
				} else {
					thetas = thetas.subRowVector( thetas.getRow(vs) ).divi( 2.0 ).addRowVector( thetas.getRow(vs) ) ; 
					f[vg] = cost( X, y, thetas.getRow(vg).transpose(), LAMBDA );
					f[vh] = cost( X, y, thetas.getRow(vh).transpose(), LAMBDA );
				}
			}

			/* test for convergence */
			double fsum = 0.0;
			for( int j=0;j<=n;j++) {
				fsum += f[j];
			}
			double favg = fsum/(n+1);
			double s = 0.0;
			for( int j=0;j<=n;j++) {
				s += Math.pow((f[j]-favg),2.0)/(n);
			}
			s = Math.sqrt(s);
			if (s < EPSILON) break;
		}	/* end main loop of the minimization */

		log.info( "Completed solution in {} iterations", itr ) ;

		/* find the index of the smallest value */
		int vs=0;
		for( int j=1;j<=n;j++) {
			if (f[j] < f[vs]) {
				vs = j;
			}
		}
		
		return thetas.getRow( vs ) ;
	}
}

