/* A program for taking a (possibly arbitrary) alignment score matrix
 * and back-calculating the implied target frequencies p_ab.
 * 
 * Doing this requires solving for a nonzero lambda in:
 *         \sum_ab f_a f_b e^{\lambda s_ab} = 1
 * and this is a good excuse to demo two methods of root-finding: 
 * bisection search and the Newton/Raphson method.
 * 
 * The program is ANSI C, and should compile on any machine with a
 * C compiler:
 *       % cc -o lambda lambda.c -lm
 *      
 * You need two datafiles. The <backgroundfile> lists the background
 * residue frequencies; it is also used to determine what alphabet
 * you're using (DNA, 4 letters; or amino acids, 20 letters).  The
 * <matrixfile> is the score matrix. The format of this is the same as
 * any BLAST scoring matrix. (Any letters that aren't in your
 * backgroundfile are ignored; this lets you read matrices that
 * include scores for IUPAC ambiguity codes.) 
 * 
 * Example formats:
 *   a <backgroundfile>:
 *      A 0.25
 *      C 0.25
 *      G 0.25
 *      T 0.25
 *      
 *   a <matrixfile>:
 *        A   C   G   T
 *     A +1  -2  -2  -2     
 *     C -2  +1  -2  -2
 *     G -2  -2  +1  -2
 *     T -2  -2  -2  +1
 *     
 * See the bottom of this file for amino acid examples you can 
 * cut out and save to files.
 * 
 * To run the program:
 *       % ./lambda <backgroundfile> <matrixfile>
 *       
 * The program prints the implied target frequencies for the matrix,
 * and some other statistics.
 * 
 *--------------------------------------------------------------------
 * 
 * Organization of the code:
 * 
 * section 1:  main(), the main body of the program.
 * 
 * section 2:  The root-finding methods, bisection() and
 *             newtonraphson(). Look here for an example of
 *             how these methods work.
 *             
 * section 3:  output routines. Boring. But if you want to 
 *             hack something else into the output, either hack
 *             main(), or hack one of the routines here.
 * 
 * section 4:  a couple of allocation routines.
 * 
 * section 5:  file format parsers. Equally boring. But if you
 *             are having trouble with file formats, look here
 *             for more format documentation.
 *             
 * section 6:  example file formats. Copies of the BLOSUM62 score
 *             matrix, and the background frequencies used in
 *             constructing BLOSUM62.
 *              
 *--------------------------------------------------------------------
 * 
 * SRE, Sat Jun 26 09:47:59 2004
 * This code is explicitly assigned to the public domain. You may
 * reuse it as you wish.
 * CVS $Id: lambda.c,v 1.7 2004/07/01 13:44:33 eddy Exp $ 
 */

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <math.h>

#define MAXABET       20	/* sets maximum alphabet size           */
#define TOLERANCE     1e-6      /* how close to 0 root-finders must get */


/* Function declarations (in order of appearance)
 */
static float val(float **s, float *f, int K, float lambda);
static float deriv(float **s, float *f, int K, float lambda);
static float bisection(float **s, float *f, int K);
static float newtonraphson(float **s, float *f, int K);
static void  output_alphalabeled_matrix(float **mx, int K, char *alphabet);
static void  output_alphalabeled_vector(float *f, int K, char *alphabet);
static float **FMX2Alloc(int rows, int cols);
static void    FMX2Free(float **mx);
static int ParseBackgroundFile(FILE *bfp, char *alphabet, int *ret_K, float *fq);
static int ParseScoreMatrixFile(FILE *mfp, char *alphabet, int K, float **mx);


int
main(int argc, char **argv)
{
  char  *backgroundfile;	
  char  *matrixfile;
  FILE  *fp;
  char   alphabet[MAXABET+1];	/* alphabet order (e.g. ACGT)      */
  int    K;			/* size of alphabet (4 or 20)      */
  int    a, b;			/* indices of two residues         */
  float  lambda;		/* the lambda we've solved for     */
  float **s;                    /* score matrix s_ab               */
  float  *f;                    /* background frequencies f_a, f_b */
  float **pab;                  /* solved target frequencies p_ab  */
  float   x;			/* generic single number solution  */
  float  *vec;			/* generic vector solution         */
  float **mx;			/* generic matrix solution         */


  /* Parse the command line.
   */
  if (argc != 3) {
    fprintf(stderr, "Usage: ./lambda <backgroundfile> <matrixfile>\n");
    exit (1);
  }
  backgroundfile = argv[1];
  matrixfile     = argv[2];
  
  /* Allocations.
   */
  if ((f   = malloc(sizeof(float) * MAXABET)) == NULL)
    { fprintf(stderr, "malloc failed\n"); exit(1); }
  if ((s   = FMX2Alloc(MAXABET, MAXABET)) == NULL)
    { fprintf(stderr, "malloc failed\n"); exit(1); }
  if ((pab = FMX2Alloc(MAXABET, MAXABET)) == NULL)
    { fprintf(stderr, "malloc failed\n"); exit(1); }
  if ((vec = malloc(sizeof(float) * MAXABET)) == NULL)
    { fprintf(stderr, "malloc failed\n"); exit(1); }
  if ((mx  = FMX2Alloc(MAXABET, MAXABET)) == NULL)
    { fprintf(stderr, "malloc failed\n"); exit(1); }


  /* Read in the background frequencies.
   * This also sets K (alphabet size; 4 or 20) and
   * the order of the alphabet (maybe ACGT or ACDEFGHIKLMNPQRSTVWY).
   */
  if ((fp = fopen(backgroundfile, "r")) == NULL) 
    { fprintf(stderr, "Failed to open background file %s for reading\n", backgroundfile); exit(1); }
  if (! ParseBackgroundFile(fp, alphabet, &K, f)) 
    { fprintf(stderr, "Failed to parse background file %s\n", backgroundfile); exit(1); }
  fclose(fp);  
  
  printf("Your input background frequencies:\n");
  output_alphalabeled_vector(f, K, alphabet);
  printf("\n");

  /* Read in the scoring matrix, which must be KxK.
   * Print it out (so user can make sure it's what they thought.)
   */
  if ((fp = fopen(matrixfile, "r")) == NULL) 
    { fprintf(stderr, "Failed to open matrix file %s for reading\n", matrixfile); exit(1);}
  if (! ParseScoreMatrixFile(fp, alphabet, K, s)) 
    { fprintf(stderr, "Failed to parse matrix file %s\n", matrixfile); exit(1); }
  fclose(fp);

  printf("Your input scoring matrix:\n");
  output_alphalabeled_matrix(s, K, alphabet);
  printf("\n");

  /* Verify the two necessary conditions for the matrix:
   * 1. There must be at least one positive score.
   * 2. The expected score must be negative.
   */
  for (a = 0; a < K; a++) 
    for (b = 0; b < K; b++)
      if (s[a][b] > 0.) goto HAVE_POSITIVE; 
  fprintf(stderr, "Score matrix has no positive score: can't solve lambda\n");
  exit(1);
 HAVE_POSITIVE: /* one of few excuses for a goto: break out of nested loop. */
  
  x = 0.;
  for (a = 0; a < K; a++)
    for (b = 0; b < K; b++)
      x += f[a] * f[b] * s[a][b];
  if (x >= 0.) {
    fprintf(stderr, "Score matrix has nonnegative expectation: can't solve lambda\n");
    exit(1);
  }

  /* Find lambda by Newton/Raphson algorithm.
   * Replace w/ bisection(s,f,K) if you want to play with the other
   * rootfinding method.
   */
  lambda = newtonraphson(s, f, K);  

  printf("Solve for lambda (Newton/Raphson method):\n");
  printf("lambda    = %f\n", lambda);
  printf("f(lambda) = %f ~= 0\n", val(s, f, K, lambda));
  printf("\n");

  /* Use lambda to backcalculate target frequencies p_ab:
   *    p_ab = f_a f_b e^{lambda s_ab}
   */
  for (a = 0; a < K; a++) 
    for (b = 0; b < K; b++) 
      pab[a][b] = f[a]*f[b]* exp(lambda * s[a][b]);
      
  printf("The implied target probabilities p_ab:\n");
  output_alphalabeled_matrix(pab, K, alphabet);
  printf("\n");

  
  /* The marginal probabilities of residues p_a in
   * the query sequence: not necessarily the same as
   * the background frequencies.
   *            p_a = \sum_b p_ab
   */
  for (a = 0; a < K; a++) {
    vec[a] = 0.;
    for (b = 0; b < K; b++) 
      vec[a] += pab[a][b];
  }
  printf("The implied marginal query composition p_a:\n");
  output_alphalabeled_vector(vec, K, alphabet);
  printf("\n");

  
  /* The conditional probabilities p(b | a) for
   * probability of target residue b given query residue a:
   *       p(b | a) = p_ab / p_a
   */
  for (a = 0; a < K; a++)
    for (b = 0; b < K; b++)
      mx[a][b] = pab[a][b] / vec[a];

  printf("Implied conditional probabilities p(b|a):\n");
  output_alphalabeled_matrix(mx, K, alphabet);
  printf("\n");


  /* Find the expected score of the score matrix:
   *    \sum_ab f_a f_b s_ab
   * Report in bits: so, x lambda / log 2.
   * (We already made sure this was negative, but now that we
   *  know lambda, we can report it in units of bits.)
   * 
   * Note: for BLOSUM62, result is -.46 bits; but header of BLOSUM62
   * file says expect = -.52. Discrepancy arises from a roundoff
   * error. matblas (the program that constructed BLOSUM62) calculated
   * the expected score using an unrounded version of s_ab, but later
   * scaled and rounded to integers to get the BLOSUM62 we know and
   * love. Same applies for discrepancy in lambda: should be .3466
   * (log 2/2), but the actual integer BLOSUM62 gives .3208.  The
   * original unrounded blosum62.sij can be found on the Net.
   */
  x = 0.;
  for (a = 0; a < K; a++)
    for (b = 0; b < K; b++)
      x += f[a] * f[b] * s[a][b];
  x *= lambda / log(2.);
  printf("Expected score  = %.4f bits\n", x);
  
  /* Find the relative entropy (average score in true alignments)
   *    H = \sum_ab p_ab s_ab
   * Report in bits.
   */
  x = 0.;
  for (a = 0; a < K; a++)
    for (b = 0; b < K; b++)
      x += pab[a][b] * s[a][b];
  x *= lambda / log(2.);	/* convert to bits */
  printf("Relative entropy = %.4f bits\n", x);


  /* Find the expected % sequence identity in homologous
   * alignments:
   */
  x = 0.;
  for (a = 0; a < K; a++)
    x += pab[a][a];
  printf("Target %% identity = %.1f\n", x*100.);
  
  FMX2Free(s);
  FMX2Free(pab);
  FMX2Free(mx);
  free(f);
  free(vec);
  exit(0);
}


/* ================================================================
 * Section 2: Solving for lambda with rootfinding methods.
 * ================================================================
 *
 * We're looking for the nonzero \lambda that satisfies:
 *        \sum_ab f_a f_b e^{\lambda s_ab} - 1  = 0
 *        
 * And if s_ab has at least one positive score, and has a negative
 * expectation, there is guaranteed to be one and only one nonzero
 * solution. 
 * 
 * This is a rootfinding problem (find x that satisfies f(x) = 0). 
 * It sets up a nice, simple example for seeing how the bisection and
 * Newton/Raphson methods work.
 * 
 * val() calculates f(lambda) for a given lambda. This is the 
 *       function that we're trying to find a root for.
 *       
 * deriv() is the first derivative f'(lambda) at a given lambda.
 *       We need first derivative info for the Newton/Raphson
 *       method.
 *       
 * bisection() implements the bisection search, and 
 * newtonraphson() implements Newton/Raphson.      
 * 
 * 
 * A good general reference on numerical root finding methods is
 * _Numerical Recipes in C_, WH Press et al., Cambridge Univ Press 1993.
 */
/* Calculate and return f(lambda):  \sum_ab f_a f_b e^{lambda s_ab} - 1
 */                                     
static float
val(float **s, float *f, int K, float lambda)
{
  int   a,b;
  float total = -1.;

  for (a = 0; a < K; a++)
    for (b = 0; b < K; b++)
      total += f[a] * f[b] * exp(lambda * s[a][b]);
  return total;
}
/* First derivative is:  \sum_ab f_a f_b s_ab e^{lambda s_ab}
 */
static float
deriv(float **s, float *f, int K, float lambda)
{
  int   a,b;
  float deriv = 0.;

  for (a = 0; a < K; a++)
    for (b = 0; b < K; b++)
      deriv += f[a] * f[b] * s[a][b] * exp(lambda * s[a][b]);
  return deriv;
}

/* The bisection method.
 *
 * The key idea of bisection is this: you bracket the root, finding an
 * x1 for which f(x1) < 0, and an x2 for which f(x2) > 0 -- so the
 * solution where f(x)=0 must lie in the interval (x1,x2).  (Here, we
 * also happen to know that x1 < x2; f(x) is an increasing function
 * around the root we care about.) Then you guess a point x3 in
 * between, x1 < x3 < x2.  If f(x3) > 0, then the solution is still to
 * our left; we reset the bracketed interval to (x1,x3). If f(x3) < 0,
 * the solution is to our right, and we reset the interval to
 * (x3,x2). We iterate this until we reach a suitably close f(x) ~ 0.
 * 
 * Bisection is very robust; it is guaranteed to find at least one
 * root, if at least one root exists.
 * 
 * Numerically, we can't expect to reach f(x) = 0 exactly. 
 * We may reach two "adjacent" x's in the computer's floating point
 * representation that bracket the root. We can either check
 * that f(x) is "close enough" (within some TOLERANCE, say 1e-5, of 0)
 * or we can directly test that x1 and x2 are adjacent floating
 * point numbers: if (x1+x2)/2 = x2 || (x1+x2)/2 == x2, this is
 * the case. We use both tests here.
 */
static float
bisection(float **s, float *f, int K)
{
  float initial_guess;
  float left_lambda, right_lambda, mid_lambda;
  float fx;
  int   iter;

  /* Our initial guess can be anything, but it doesn't
   * hurt to be close.
   */
  initial_guess = 0.2;

  /* Find lambda that gives a result f(lambda) < 0;
   * this is the left side of our initial bracket.
   */
  left_lambda = initial_guess;			
  for (iter = 0; iter < 100; iter++) {
    if (val(s, f, K, left_lambda) < 0) break;
    left_lambda /= 2.;
  }
  if (iter == 100) 
    { fprintf(stderr, "Failed to find left bracket\n"); exit(1); }
  
  /* Find lambda that gives a result of f(lambda) > 0;
   * this is the right side of our initial bracket.
   */
  right_lambda = initial_guess;
  for (iter = 0; iter < 100; iter++) {
    if (val(s, f, K, right_lambda) > 0) break;
    right_lambda *= 2.;
  }
  if (iter == 100) 
    { fprintf(stderr, "Failed to find right bracket\n"); exit(1); }

  /* Now, the iterative bisection search.
   */
  for (iter = 0; iter < 100; iter++) {
    mid_lambda = (left_lambda + right_lambda) / 2;
    /* check if interval is as tight as numerically possible: */
    if (mid_lambda == left_lambda || mid_lambda == right_lambda) break; 
    fx = val(s, f, K, mid_lambda);
    /* check if we're suitably close to f(lambda) == 0 */
    if (fabs(fx) < TOLERANCE) break; 
    /* narrow the bracket: */
    if (fx > 0) right_lambda = mid_lambda;
    else        left_lambda  = mid_lambda;
  }
  if (iter == 100) 
    { fprintf(stderr, "Bisection search failed\n"); exit(1); }
  
  return mid_lambda;
}

/* The Newton/Raphson method.
 * 
 * The basic idea is that for any guess of x, the first derivative f'x
 * should tell us which direction the root is in; and we can even make
 * a good guess where, by assuming f(x) is a straight line defined by
 * the slope f'(x). Our new guess for x is x - f(x)/f'(x). (If you
 * don't see why, draw the situation on a graph; it should be easy to
 * see geometrically.) We can iterate this until we find the root.
 * 
 * Newton/Raphson is not robust in general; there are pathological
 * f(x)'s that make it blow up. We have to be careful that our
 * f(x) is amenable to Newton/Raphson. Here, f(lambda) is well behaved.
 * The only trick is that we have two roots (one of them is lambda=0,
 * and one of them is the lambda we want to find), so we don't
 * want Newton/Raphson to find the lambda=0 solution. It's sufficient
 * to make sure that we start with an initial guess to the right
 * of our desired root, where f(lambda) > 0. 
 * 
 * Again, numerical convergence must be tested, finding when 
 * f(lambda) is within TOLERANCE (1e-5 or so) of 0.
 */
static float
newtonraphson(float **s, float *f, int K)
{
  float lambda, newlambda;
  int   iter;
  float fx, dfx;
  
  /* There's two zeros, not one: lambda=0 is a solution.
   * To find the nonzero zero (ahem), we've got to make sure 
   * we start to the right of it: where the function is positive.
   * Start with anything, then move right until fx > 0.
   */
  lambda = 0.1;
  for (iter = 0; iter < 100; iter++) {
    if (val(s, f, K, lambda) > 0.) break;
    lambda *= 2.;
  }
  if (iter == 100) { 
    fprintf(stderr, "failed to find a start pt for newton/raphson\n");
    exit(1);
  }

  /* Now, starting from there, come back in with the Newton/Raphson
   * algorithm. At least in theory, because we know what f(x) looks
   * like, this should be well-behaved, smoothly converging to the
   * solution.
   */
  for (iter = 0; iter < 100; iter++) {
    fx = val(s, f, K, lambda);
    if (fabs(fx) < TOLERANCE) break;   /* success! */
    dfx = deriv(s, f, K, lambda);
    newlambda = lambda - fx / dfx; /* This update defines Newton/Raphson */
    if (newlambda <= 0) newlambda = 0.000001; /* this shouldn't happen */
    lambda = newlambda;
  }
  if (iter == 100) 
    { fprintf(stderr, "Newton/Raphson search failed\n"); exit(1); }
  return lambda;
}



/* ================================================================
 * Section 3:           Output routines.
 * ================================================================
 */
/* Output an alphabet-labeled square matrix of floating point #'s.
 */
static void
output_alphalabeled_matrix(float **mx, int K, char *alphabet)
{
  int a,b;

  printf("  "); 
  for (b = 0; b < K; b++) 
    printf("       %c ", alphabet[b]);
  printf("\n");
  for (a = 0; a < K; a++) {
    printf("%c ", alphabet[a]);
    for (b = 0; b < K; b++) 
      printf("%8.4f ", mx[a][b]);
    printf("\n");
  }
}
/* Output an alphabet-labeled vector of floating point #'s.
 */
static void
output_alphalabeled_vector(float *f, int K, char *alphabet)
{
  int a;

  printf("  "); 
  for (a = 0; a < K; a++) printf("       %c ", alphabet[a]);
  printf("\n");
  printf("  "); 
  for (a = 0; a < K; a++) printf("%8.4f ", f[a]);
  printf("\n");  
}


/* ================================================================
 * Section 4:          Allocation routines.
 * ================================================================
 */
static float **
FMX2Alloc(int rows, int cols)
{
  float **mx;
  int     r;
  if ((mx = (float **) malloc(sizeof(float *) * rows)) == NULL)
    return NULL;
  if ((mx[0] = (float *)  malloc(sizeof(float) * rows * cols)) == NULL)
    { free(mx); return NULL; }
  for (r = 1; r < rows; r++)
    mx[r] = mx[0] + r*cols;
  return mx;
}
static void
FMX2Free(float **mx)
{
  free(mx[0]);
  free(mx);
}



/* ================================================================
 * Section 5:          File parsers.
 * ================================================================
 */
/* Function:  ParseBackgroundFile()
 * Incept:    SRE, Tue Jun 22 15:03:47 2004 [St. Louis]
 *
 * Purpose:   Parse a file of background residue frequencies.
 *
 * File format: 
 *            Blank lines are ignored. Lines w/ # as their 
 *            first non-whitespace are comments, and are also
 *            ignored.
 *            
 *            Expects lines in format:
 *            <residue>    <frequency of residue>
 *            
 *            Example:
 *            -----------------------
 *            A   0.33
 *            C   0.17
 *            G   0.17
 *            T   0.33
 *            -----------------------
 *            
 *            This gets parsed into fq[], in the residue
 *            order specified by alphabet. The parser also
 *            stores the alphabet[], and the size of the
 *            alphabet K. 
 *            
 *            Example:
 *            -----------------------
 *            float fq[4];
 *            char  alphabet[MAXABET+1];
 *            int   K;
 *            FILE *bfp;
 *            
 *            bfp = fopen("datafile", "r");
 *            ParseBackgroundFile(bfp, alphabet, &K, fq);
 *            fclose(bfp);
 *            -----------------------
 *            
 *            after this, fq[0] is f_A; fq[1] is f_C; etc;
 *            K=4; and alphabet="ACGT".
 *            
 * Args:      bfp      - open file ptr for reading.
 *            alphabet - RETURN: residue alphabet, in order of storage in fq;
 *                       caller allocates for MAXABET+1, MAXABET is the maximum
 *                       alphabet size (20).
 *            ret_K    - RETURN: # of residues in alphabet
 *            fq       - RETURN: frequencies, 0..K-1
 *                       Caller allocates this, and is responsible
 *                       for free'ing it.
 *
 * Returns:   1 on success; 0 on failure.
 *
 * Xref:      STL8 p.50
 */
static int
ParseBackgroundFile(FILE *bfp, char *alphabet, int *ret_K, float *fq)
{
  char  buffer[512];		/* input buffer for lines from fp */  
  char *s1;			/* tmp ptr into buffer, field 1 (residue) */
  char *s2;			/* tmp ptr into buffer, field 2 (frequency) */
  int   i;
  int   n;
  float sum;
  
  n = 0;
  while (fgets(buffer, 512, bfp) != NULL) 
    {
      if ((s1 = strtok(buffer, " \t\n")) == NULL) continue; /* blank  */
      if (*s1 == '#')                             continue; /* comment*/
      if ((s2 = strtok(NULL,   " \t\n")) == NULL) return 0; /* oops, eol*/
      if (n >= MAXABET)                           return 0; /* out of space */

      alphabet[n] = *s1;
      fq[n]       = atof(s2);
      n++;
    }
  alphabet[n] = '\0';
  *ret_K = n;

  /* Make sure it looks like a remotely plausible frequency
   * distribution (it should sum to one); if it's at all close,
   * normalize it.
   */
  sum = 0.;
  for (i = 0; i < n; i++) sum += fq[i];
  if (sum < 0.9 || sum > 1.1) return 0;	 
  for (i = 0; i < n; i++) fq[i] /= sum;

  return 1;
}

/* Function:  ParseScoreMatrixFile()
 * Incept:    SRE, Tue Jun 22 13:17:16 2004 [St. Louis]
 *
 * Purpose:   Takes an open score matrix file, and parses
 *            it into an allocated matrix "mx", in the
 *            order given by symbols in "alphabet". 
 *
 *            Scores for symbols other than those in "alphabet"
 *            are ignored. (For instance, score matrices often
 *            include scores for IUPAC ambiguity codes and '*',
 *            and maybe we only want to keep unambiguous standard
 *            symbols like ACGT.)
 *            
 *            Allows up to 30 rows/cols in the matrix.
 *            
 * Format of datafile:
 *            Lines starting with '#' as their first non-blank
 *            character are considered to be comments, and are
 *            ignored. Blank lines are also ignored.
 *            
 *            First data line is a header with residue symbols,
 *            labeling the columns. The number of symbols in
 *            the score matrix is determined from this line.
 *            
 *            Next "nsymbols" valid lines are the data lines
 *            of the matrix. They contain either nsymbols or
 *            nsymbols+1 fields, because they are allowed to
 *            have a leading residue label for the row (this
 *            label, if present, is ignored). These rows
 *            must be in the same order that the columns were 
 *            in.
 *            
 *            Example:
 *            
 *            --------------------------------
 *                A    C    G   T   *
 *            A   4   -2   -2  -2  -4
 *            C  -2    4   -2  -2  -4
 *            G  -2   -2    4  -2  -4
 *            T  -2   -2   -2   4  -4
 *            *  -4   -4   -4  -4  -4
 *            --------------------------------
 *            
 *            This has nsymbols=5.
 *                
 *            Example of parsing it is:
 *            
 *            ---------------------------------
 *            float mx[4][4];
 *            FILE *mfp;
 *            
 *            mfp = fopen("datafile", "r");
 *            ParseScoreMatrixFile(mfp, "ATGC", 4, mx);
 *            fclose(mfp);
 *            ---------------------------------
 *            
 *            now mx[1][1] is the score for T/T;
 *                mx[2][2] is the score for G/G; etc.
 *               (that is, the rows/cols are in the order
 *                specified by "ATGC", rearranged from how
 *                they were in the file.)
 *
 * Args:      mfp       - open score matrix file for reading;
 *                        Caller must open; caller must close.
 *            alphabet  - which letters are to be parsed, in what order;
 *                        e.g. "ACGT"
 *            K         - # of residues in the parsed alphabet.
 *            mx        - RETURN: the score matrix. 
 *                        Caller must allocate this KxK; and caller
 *                        is responsible for free'ing it.
 *
 * Returns:   1 on success; 0 on failure.
 *
 * Xref:      STL8 p.50.
 */
static int
ParseScoreMatrixFile(FILE *mfp, char *alphabet, int K, float **mx)
{
  char  buffer[512];		/* input buffer for lines from fp */
  int   order[30];		/* order of fields, from header   */
  char *sptr;                   /* tmp ptr into the buffer        */
  char *s2;			/* tmp ptr into alphabet          */
  int   nsymbols;		/* total # of symbols in the mx   */
  int   row,col;		/* counters for rows, columns     */
  int   i,n;

  for (i = 0; i < 30; i++) 
    order[i] = -1;

  /* Look for the first non-blank, non-comment line in the file.
   * It gives single-letter codes in the order the PAM matrix
   * is arrayed in the file. 
   *  (fgets() and strtok() are both deprecated in real C
   *   applications; only used here for a simple ANSI C example.)
   */
  do {
    if (fgets(buffer, 512, mfp) == NULL) return 0;
  } while ((sptr = strtok(buffer, " \t\n")) == NULL || *sptr == '#');
  
  /* Parse that line; figure out which rows/columns we're going
   * to store, and in what order. (This lets the caller request only
   * unambiguous, standard codes like a 4x4 matrix indexed for "ACGT",
   * even if the matrix file contains scores for IUPAC ambiguity
   * codes, and/or w/ symbols in a different order.)
   */
  i = 0;
  do {
    s2 = strchr(alphabet, *sptr); /* is this a residue we'll store? */
    if (s2 != NULL) order[i] = (int) (s2-alphabet);  /* yes */
    i++;
  } while ((sptr = strtok(NULL, " \t\n")) != NULL);
  nsymbols = i;

  /* Doublecheck that we're going to store a KxK matrix.
   */
  n=0;
  for (i = 0; i < 30; i++) if (order[i] != -1) n++;
  if (n != K) return 0;

  /* The next nsymbols rows are the score matrix.
   * The first field of each may or may not be a letter, depending
   * on what format the matrix is in; if it's a letter, ignore it.
   */
  row = 0;
  do {
    if (fgets(buffer, 512, mfp) == NULL)          return 0; /* oops, eof  */
    if ((sptr = strtok(buffer, " \t\n")) == NULL) continue; /* blank line */
    if (*sptr == '#')                             continue; /* comment    */
    
    if (isalpha(*sptr)) {	                            /* skip char  */
      if ((sptr = strtok(NULL, " \t\n")) == NULL) return 0; /* bad line   */
    }

    col = 0;
    do {			/* parse each field on the line */
      if (sptr == NULL) return 0;                            /* oops, eol */
      if (order[row] != -1 && order[col] != -1)              /* a keeper? */
	mx[order[row]][order[col]] = atof(sptr);             /* yes.      */
      sptr = strtok(NULL, " \t\n");
      col++;
    } while (col < nsymbols);

    row++;
  } while (row < nsymbols);

  return 1;
}




/* ================================================================
 * Section 6:         Example data files.
 * ================================================================
 */

/* ---------------------------------------------------------------- 
 * Example amino acid score matrix file (BLOSUM62)
 * Cut and save in your own file.
 *
 * ---------------------- <snip> ---------------------------------- 
#  Matrix made by matblas from blosum62.iij
#  * column uses minimum score
#  BLOSUM Clustered Scoring Matrix in 1/2 Bit Units
#  Blocks Database = /data/blocks_5.0/blocks.dat
#  Cluster Percentage: >= 62
#  Entropy =   0.6979, Expected =  -0.5209
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4 
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4 
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4 
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4 
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4 
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4 
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4 
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4 
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4 
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4 
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4 
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4 
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4 
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4 
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4 
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4 
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4 
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4 
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4 
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4 
B -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4 
Z -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4 
X  0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4 
* -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1 
 * ---------------------- <snip> ---------------------------------- 
 *
 */ 
 
/* Example background frequency file.
 * [These were the background frequencies used to construct BLOSUM62;
 *  from blosum62.out, the matblas output file from 1992.]
 * Cut and save to a file.
 * ---------------------- <snip> ---------------------------------- 
A 0.074
C 0.025
D 0.054
E 0.054
F 0.047
G 0.074
H 0.026
I 0.068
L 0.099
K 0.058
M 0.025
N 0.045
P 0.039
Q 0.034
R 0.052
S 0.057
T 0.051
V 0.073
W 0.013
Y 0.034
 * ---------------------- <snip> ---------------------------------- 
 *
 */ 
