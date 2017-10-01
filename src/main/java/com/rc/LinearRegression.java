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

	public DoubleMatrix solve( DoubleMatrix x,  DoubleMatrix y, double learningRate, double lambda, double threshold, int maxIterations ) {
		DoubleMatrix I = DoubleMatrix.ones( x.rows ) ;
		DoubleMatrix X = DoubleMatrix.concatHorizontally(I, x) ;

		theta = DoubleMatrix.zeros( X.columns ) ;

		NMSimplex(X, y, theta.data, 1e-12 );

		return theta ;
	}


	public double cost( DoubleMatrix X, DoubleMatrix y, DoubleMatrix theta, double lambda ) {
		DoubleMatrix S = X.mmul( theta ).subi( y ) ;		
		double j1 = S.muli(S).sum() ;

		DoubleMatrix t = theta.mul( theta ).muli( lambda ) ;
		double cost = ( t.sum() - t.get(0) + j1 ) / ( 2 * y.length ) ;

		return cost ;
	}


	private DoubleMatrix gradients( DoubleMatrix X, DoubleMatrix y, double lambda ) {

		//		DoubleMatrix G1 = X.mmul( theta ).subi( y ).transpose().mmul( X ) ;
		DoubleMatrix S = X.mmul( theta ).subi( y ) ;		
		DoubleMatrix G1 = X.transpose().mmul( S ) ;

		DoubleMatrix G2 = theta.mul( lambda ) ;
		G2.put(0, 0) ;

		return G2.addi( G1 ).divi( y.length ) ;
	}


	static final int MAX_IT = 1000;      	/* maximum number of iterations */
	static final double ALPHA  = 1.5;       /* reflection coefficient */
	static final double BETA   = 0.5;       /* contraction coefficient */
	static final double GAMMA  = 2.0;       /* expansion coefficient */
	static final double LAMBDA = 1e-6;    	/* regularization coefficient */

	public void NMSimplex(DoubleMatrix X, DoubleMatrix y, double startIn[], double EPSILON ) {

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
		DoubleMatrix start = new DoubleMatrix( startIn ) ; 
		for( int i=0;i<n;i++) {
			v2.putRow(0, start) ;
		}

		for( int i=1;i<=n;i++) {
			v2.putRow( i, start.add(qn).put( i-1, startIn[i-1]+pn) ) ;
		}
		
		for( int j=0;j<=n;j++) {
			f[j] = cost( X, y, v2.getRow(j).transpose(), LAMBDA ) ;
		}

		
		/* begin the main loop of the minimization */
		for( itr=0 ; itr<MAX_IT ; itr++ ) {     
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
		System.out.println( "Completed " + itr  + " iterations." ) ;

		/* find the index of the smallest value */
		vs=0;
		for( int j=0;j<=n;j++) {
			if (f[j] < f[vs]) {
				vs = j;
			}
		}

		for( int j=0;j<n;j++) {
			theta = v2.getRow( vs ) ;
		}
	}

	
	public void NMSimplexOrig(DoubleMatrix X, DoubleMatrix y, double start[], double EPSILON ) 
	{

		int n = X.columns ;
		double scale = 1 ;
		double v[][] = new double[n+1][n];
		double f[]   = new double[n+1];
		double vr[]  = new double[n];
		double ve[]  = new double[n];
		double vc[]  = new double[n];
		double vm[]  = new double[n];
//		double fr;      /* value of function at reflection point */
//		double fe;      /* value of function at expansion point */
//		double fc;      /* value of function at contraction point */
		double fsum,favg,s,cent;
		int vs;         /* vertex with smallest value */
		int vh;         /* vertex with next smallest value */
		int vg;         /* vertex with largest value */
		int i,j,m,row = 0;
		int itr;	      /* track the number of iterations */

		/* create the initial simplex */
		/* assume one of the vertices is 0,0 */

		double pn = scale*(Math.sqrt(n+1)-1+n)/(n*Math.sqrt(2));
		double qn = scale*(Math.sqrt(n+1)-1)/(n*Math.sqrt(2));

		for (i=0;i<n;i++) {
			v[0][i] = start[i];
		}

		for (i=1;i<=n;i++) {
			for (j=0;j<n;j++) {
				if (i-1 == j) {
					v[i][j] = pn + start[j];
				}
				else {
					v[i][j] = qn + start[j];
				}
			}
		}
		

		/* find the initial function values */
		DoubleMatrix v2 = new DoubleMatrix( v ) ;
		
		//cost( X, y, v2.transpose(), 0 ) ;
		for (j=0;j<=n;j++) {
			//f[j] = objf.evalObjfun(v[j]);
			f[j] = cost( X, y, v2.getRow(j).transpose(), 0 ) ;
		}

		
		/* begin the main loop of the minimization */
		for (itr=0;itr<MAX_IT;itr++) {     
			/* find the index of the largest value */
			vg=0;
			for (j=0;j<=n;j++) {
				if (f[j] > f[vg]) {
					vg = j;
				}
			}
			

			/* find the index of the smallest value */
			vs=0;
			for (j=0;j<=n;j++) {
				if (f[j] < f[vs]) {
					vs = j;
				}
			}

			/* find the index of the second largest value */
			vh=vs;
			for (j=0;j<=n;j++) {
				if (f[j] > f[vh] && f[j] < f[vg]) {
					vh = j;
				}
			}

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

			DoubleMatrix vm2 = v2.columnSums().subi( v2.getRow(vg) ).div( n ) ; 
			
			/* Average all points on the simplex - except the largest to get a new point */
			for (j=0;j<=n-1;j++) {
				cent=0.0;
				for (m=0;m<=n;m++) {
					if (m!=vg) {
						cent += v[m][j];
					}
				}
				vm[j] = cent/n;
			}

			DoubleMatrix vr2 = vm2.sub( v2.getRow(vg) ).muli( ALPHA ).addi( vm2 ).transpose() ;
			/* reflect vg to new vertex vr */
			for (j=0;j<=n-1;j++) {
				vr[j] = vm[j]+ALPHA*(vm[j]-v[vg][j]);
			}

//			fr = objf.evalObjfun(vr);
			
			DoubleMatrix vr3 = new DoubleMatrix( vr ) ;
			double fr = cost( X, y, vr2, 0 ) ;

			if( fr < f[vh] && fr >= f[vs]) {
				for (j=0;j<=n-1;j++) {
					v[vg][j] = vr[j];
				}
				v2.putRow( vg, vr2) ;
				f[vg] = fr;
			}

			if( fr < f[vh] && fr >= f[vs]) {
				v2.putRow(vg, vr2);
				f[vg ] = fr ;
			}
			
			/* investigate a step further in this direction */
			if ( fr <  f[vs]) {
				for (j=0;j<=n-1;j++) {
					/*ve[j] = GAMMA*vr[j] + (1-GAMMA)*vm[j];*/
					ve[j] = vm[j]+GAMMA*(vr[j]-vm[j]);
				}
				DoubleMatrix ve2 = vm2.sub( vr2 ).muli( GAMMA ).addi( vm2 ).transpose() ;

//				fe = objf.evalObjfun(ve);
				DoubleMatrix ve3 = new DoubleMatrix( ve ) ;
				double fe = cost( X, y, ve2, 0 );
				if (fe < f[vs] ) {
					for (j=0;j<=n-1;j++) {
						v[vg][j] = ve[j];
					}
					f[vg] = fe;
					v2.putRow( vg, ve2 ) ;
				}
				else {
					for (j=0;j<=n-1;j++) {
						v[vg][j] = vr[j];
					}
					f[vg] = fr;
					v2.putRow( vg, vr2 ) ;
				}
			}

			DoubleMatrix vc2 ;
			/* check to see if a contraction is necessary */
			if (fr >= f[vh]) {
				if (fr < f[vg] && fr >= f[vh]) {
					/* perform outside contraction */
					for (j=0;j<=n-1;j++) {
						/*vc[j] = BETA*v[vg][j] + (1-BETA)*vm[j];*/
						vc[j] = vm[j]+BETA*(vr[j]-vm[j]);
					}
					vc2 = vr2.sub( vm2 ).muli( BETA ).addi( vm2 ) ;
				}
				else {
					/* perform inside contraction */
					for (j=0;j<=n-1;j++) {
						/*vc[j] = BETA*v[vg][j] + (1-BETA)*vm[j];*/
						vc[j] = vm[j]-BETA*(vm[j]-v[vg][j]);
					}
					vc2 = vm2.sub( v2.getRow(vg) ).muli( -BETA ).addi( vm2 ).transpose() ;
				}
				
//				fc = objf.evalObjfun(vc);
				DoubleMatrix vc3 = new DoubleMatrix( vc ) ;
				double fc = cost( X, y, vc2, 0 );

				if (fc < f[vg]) {
					for (j=0;j<=n-1;j++) {
						v[vg][j] = vc[j];
					}
					v2.putRow( vg, vc2 ) ;
					f[vg] = fc;
				} else {
					for (row=0;row<=n;row++) {
						if (row != vs) {
							for (j=0;j<=n-1;j++) {
								v[row][j] = v[vs][j]+(v[row][j]-v[vs][j])/2.0;
							}
						}
					}
					DoubleMatrix v3 = v2.subRowVector( v2.getRow(vs) ).divi( 2.0 ).addRowVector( v2.getRow(vs) ) ; 
//					f[vg] = objf.evalObjfun(v[vg]);
					DoubleMatrix f2 = new DoubleMatrix( v[vg] ) ;
					f[vg] = cost( X, y, f2, 0 );

//					f[vh] = objf.evalObjfun(v[vh]);
					DoubleMatrix f3 = new DoubleMatrix( v[vh] ) ;
					f[vh] = cost( X, y, f3, 0 );
				}
			}

			/* test for convergence */
			fsum = 0.0;
			for (j=0;j<=n;j++) {
				fsum += f[j];
			}
			favg = fsum/(n+1);
			s = 0.0;
			for (j=0;j<=n;j++) {
				s += Math.pow((f[j]-favg),2.0)/(n);
			}
			s = Math.sqrt(s);
			if (s < EPSILON) break;
		}
		System.out.println( "Completed " + itr  + " iterations." ) ;
		/* end main loop of the minimization */

		/* find the index of the smallest value */
		vs=0;
		for (j=0;j<=n;j++) {
			if (f[j] < f[vs]) {
				vs = j;
			}
		}

		for (j=0;j<n;j++) {
//			System.out.format("%e\n",v[vs][j]);
			start[j] = v[vs][j];
			theta.put( j, v[vs][j] ) ;
		}
	}

}

