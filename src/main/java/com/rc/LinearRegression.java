package com.rc;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;

public class LinearRegression {

	private DoubleMatrix theta ;

	public DoubleMatrix solveFast( DoubleMatrix x,  DoubleMatrix y ) {
		if( x.isSquare() && y.length==x.rows ) {
		//	return Solve.solve(x, y) ;
		}

		DoubleMatrix I = DoubleMatrix.ones( x.rows ) ;
		DoubleMatrix X = DoubleMatrix.concatHorizontally(I, x) ;
		return Solve.solveLeastSquares(X, y) ;		
	}

	public DoubleMatrix solve( DoubleMatrix x,  DoubleMatrix y, double lambda, double threshold, int maxIterations ) {
		DoubleMatrix I = DoubleMatrix.ones( x.rows ) ;
		DoubleMatrix X = DoubleMatrix.concatHorizontally(I, x) ;

		theta = DoubleMatrix.zeros( X.columns ) ;

		return nelderMead(X, y, theta, 1e-12, lambda, 1000 ).transpose();
	}


	public double cost( DoubleMatrix X, DoubleMatrix y, DoubleMatrix theta, double lambda ) {
		DoubleMatrix S = X.mmul( theta ).subi( y ) ;		
		double j1 = S.muli(S).sum() ;

		DoubleMatrix t = theta.mul( theta ).muli( lambda ) ;
		double cost = ( t.sum() - t.get(0) + j1 ) / ( 2 * y.length ) ;

		return cost ;
	}



	static final int MAX_IT = 1000;      	/* maximum number of iterations */
	static final double ALPHA  = 1.5;       /* reflection coefficient */
	static final double BETA   = 0.5;       /* contraction coefficient */
	static final double GAMMA  = 2.0;       /* expansion coefficient */

	public DoubleMatrix nelderMead(DoubleMatrix X, DoubleMatrix y, DoubleMatrix start, double EPSILON, double LAMBDA, int maxIterations ) {

		int n = X.columns ;
		double scale = 1 ;
		double f[]   = new double[n+1];

		int vs;         /* vertex with smallest value */
		int vh;         /* vertex with next smallest value */
		int vg;         /* vertex with largest value */
		int itr;	    /* track the number of iterations */

		double pn = scale*(Math.sqrt(n+1)-1+n)/(n*Math.sqrt(2));
		double qn = scale*(Math.sqrt(n+1)-1)/(n*Math.sqrt(2));

		DoubleMatrix v2 = new DoubleMatrix( n+1, n ) ;
		for( int i=0;i<n;i++) {
			v2.putRow(0, start) ;
		}

		for( int i=1;i<=n;i++) {
			v2.putRow( i, start.add(qn).put( i-1, start.get(i-1)+pn) ) ;
		}
		
		for( int j=0;j<=n;j++) {
			f[j] = cost( X, y, v2.getRow(j).transpose(), LAMBDA ) ;
		}

		
		/* begin the main loop of the minimization */
		for( itr=0 ; itr<maxIterations ; itr++ ) {     
			// vg = largest
			// vh = 2nd largest
			// vs = smallest
			DoubleMatrix f22 = new DoubleMatrix( f ) ;
			vs = f22.argmin() ;
			vg = f22.argmax() ;
			double tmp = f22.get( vg ) ;
			f22.put( vg, Double.MIN_VALUE ) ;
			vh = f22.argmax() ;
			f22.put( vg, tmp ) ;

			DoubleMatrix vm = v2.columnSums().subi( v2.getRow(vg) ).div( n ) ; 
			
			DoubleMatrix vr2 = vm.sub( v2.getRow(vg) ).muli( ALPHA ).addi( vm ).transpose() ;
			double fr = cost( X, y, vr2, LAMBDA ) ;

			if( fr < f[vh] && fr >= f[vs]) {
				v2.putRow(vg, vr2);
				f[vg ] = fr ;
			}
			
			/* investigate a step further in this direction */
			if ( fr <  f[vs]) {
				DoubleMatrix ve2 = vm.sub( vr2 ).muli( GAMMA ).addi( vm ).transpose() ;

				double fe = cost( X, y, ve2, LAMBDA );
				if (fe < f[vs] ) {
					f[vg] = fe;
					v2.putRow( vg, ve2 ) ;
				}
				else {
					f[vg] = fr;
					v2.putRow( vg, vr2 ) ;
				}
			}

			DoubleMatrix vc2 ;
			/* check to see if a contraction is necessary */
			if (fr >= f[vh]) {
				if (fr < f[vg] && fr >= f[vh]) {
					/* perform outside contraction */
					vc2 = vr2.sub( vm ).muli( BETA ).addi( vm ) ;
				}
				else {
					/* perform inside contraction */
					vc2 = vm.sub( v2.getRow(vg) ).muli( -BETA ).addi( vm ).transpose() ;
				}
				
				double fc = cost( X, y, vc2, LAMBDA );

				if (fc < f[vg]) {
					v2.putRow( vg, vc2 ) ;
					f[vg] = fc;
				} else {
					v2 = v2.subRowVector( v2.getRow(vs) ).divi( 2.0 ).addRowVector( v2.getRow(vs) ) ; 
					f[vg] = cost( X, y, v2.getRow(vg).transpose(), LAMBDA );
					f[vh] = cost( X, y, v2.getRow(vh).transpose(), LAMBDA );
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

		/* find the index of the smallest value */
		vs=0;
		for( int j=0;j<=n;j++) {
			if (f[j] < f[vs]) {
				vs = j;
			}
		}
		
		return v2.getRow( vs ) ;
	}
}

