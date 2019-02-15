#include <RcppArmadillo.h>

// This depends statement is needed to tell R where to find the
// additional header files.
//
// [[Rcpp::depends(RcppArmadillo)]]
//

using namespace Rcpp;

// FUNCTION DECLARATIONS
// ---------------------
arma::mat outerAddition     (const arma::vec& a, const arma::vec& b);
double dot2                 (NumericVector x, NumericVector y);
void updateBj               (const arma::vec& xj, arma::mat& Rt,
                             arma::mat& Phit, arma::vec& pi,
                             const arma::vec& s1inv, const arma::vec& s2inv,
                             arma::mat& Bt, arma::vec& a, arma::vec& entsum,
                             double sigma2, int j,
                             double wj, double stepsize, double epstol,
                             int p, int D);

// FUNCTION DEFINITIONS
// --------------------
// [[Rcpp::export]]
List caisa_multivariate_reg(const arma::mat& X,
                            const arma::vec& w, const arma::vec& sa2,
                            arma::mat& Phit, arma::vec& pi,
                            arma::mat& Bt, arma::mat& Rt,
                            double sigma2,
                            int maxiter, double convtol, double epstol,
                            double stepsize, bool updatesigma,
                            bool verbose) {
  
  // ---------------------------------------------------------------------
  // DEFINE SIZES
  // ---------------------------------------------------------------------
  int n                  = X.n_rows;
  int p                  = X.n_cols;
  int D                  = Bt.n_rows;
  int K                  = sa2.n_elem;
  
  // ---------------------------------------------------------------------
  // PRECALCULATE
  // ---------------------------------------------------------------------
  const arma::mat S1inv  = 1 / outerAddition(sa2, 1/w);
  const arma::mat S2inv  = 1 / outerAddition(1/sa2, w);
  
  
  // ---------------------------------------------------------------------
  // INITIALIZE
  // ---------------------------------------------------------------------
  int iter               = 0;
  int j;
  arma::vec varobj(maxiter);
  arma::vec a(p);
  arma::vec entsum(p);
  arma::vec piold;
  
  // ---------------------------------------------------------------------
  // START LOOP : CYCLE THROUGH COORDINATE ASCENT UPDATES
  // ---------------------------------------------------------------------
  for (iter = 0; iter < maxiter; iter++) {
    
    piold              = pi;
    
    // ---------------------------------------------------------------------
    // RUN COORDINATE ASCENT UPDATES : INDEX 1 - INDEX P
    // ---------------------------------------------------------------------
    for (j = 0; j < p; j++){
      
      updateBj(X.col(j), Rt, Phit, pi, S1inv.col(j), S2inv.col(j), Bt, a, entsum, sigma2,
               j, w(j), stepsize, epstol, p, D);
      
    }
    
    // ---------------------------------------------------------------------
    // CALCULATE VARIATIONAL OBJECTIVE
    // ---------------------------------------------------------------------
    varobj(iter)          = dot(Rt,Rt)/2 - sum(square(Bt) * w)/2 + sum(a)/2;
    
    if (updatesigma)
      sigma2              = 2 * varobj(iter) / n / D;
    
    varobj(iter)          = varobj(iter) / sigma2 + log(2*PI*sigma2)/2 * n * D -
                            dot(pi, log(pi + epstol)) * p * D + sum(entsum);
    
    for (j = 1; j < K; j++){
      varobj(iter)       +=  pi(j) * log(sa2(j)) * p * D / 2 - dot( Phit.row(j), log(S2inv.row(j)) ) * D / 2;
    }
    
    // ---------------------------------------------------------------------
    // CHECK CONVERGENCE
    // ---------------------------------------------------------------------
    if (max(abs(pi - piold)) < convtol * K) {
      iter++;
      break;
    }
    
    if (iter > 0) {
      if (varobj(iter) > varobj(iter - 1)){
        iter++;
        break;
      }
    }
  }
  
  // ---------------------------------------------------------------------
  // RETURN VALUES
  // ---------------------------------------------------------------------
  
  return List::create(Named("B")       = Bt.t(),
                      Named("sigma2")  = sigma2,
                      Named("pi")      = pi,
                      Named("Phit")    = Phit,
                      Named("iter")    = iter,
                      Named("varobj")  = varobj.subvec(0,iter-1));
}

void updateBj          (const arma::vec& xj, arma::mat& Rt,
                        arma::mat& Phit, arma::vec& pi,
                        const arma::vec& s1inv, const arma::vec& s2inv,
                        arma::mat& Bt, arma::vec& a, arma::vec& entsum,
                        double sigma2, int j,
                        double wj, double stepsize, double epstol,
                        int p, int D) {
  
  pi                   += -Phit.col(j) / p;
  arma::vec c           = (Rt * xj) / wj + Bt.col(j);

  // A : likelihood matrix
  arma::mat A           = -(pow(c,2) / (2 * sigma2)) * s1inv.t();
  A.each_row()         += log(s1inv.t())/2;
  A.each_col()         -= max(A,1);
  A                     = exp(A);

  // A : component posterior probability
  arma::vec u           = 1 / (A * pi);
  arma::vec g           = A.t() * u / p + D;
  A.each_row()         %= (pi % pow(.5 + stepsize * g, 2)).t();
  A                     = normalise(A, 1, 1);

  // cpm : component posterior mean
  arma::mat cpm         = wj * c * s2inv.t();
  Rt                   += Bt.col(j) * xj.t();
  
  // update phij and pi
  Phit.col(j)           = mean(A.t(), 1);
  pi                   += Phit.col(j) / p;
  
  // update betaj = j-th row of B
  Bt.col(j)             = sum(cpm % A, 1);
  
  // update residual matrix R
  Rt                   -= Bt.col(j) * xj.t();
  
  
  //
  a(j)                  = dot(sum(A % square(cpm),0), 1/(s2inv + epstol));
  entsum(j)             = dot(A, log(A + epstol));
  
  return;
}

arma::mat outerAddition    (const arma::vec& a, const arma::vec& b) {
  arma::mat A(a.n_elem, b.n_elem);
  A.fill(0);
  A.each_row() += b.t();
  A.each_col() += a;
  return A;
}
