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
void updateBj               (const arma::vec& xj, arma::mat& Rt,
                             arma::mat& Phit, arma::vec& pi, const arma::vec& sa2,
                             const arma::vec& s1inv, const arma::vec& s2inv,
                             arma::mat& Bt, arma::vec& a, arma::vec& entsum,
                             arma::vec& B2, int j,
                             double wj, double uj, double& sigma2,
                             double stepsize, double epstol,
                             int p, int D);
void updateFactor           (arma::mat& X, const arma::mat& Y,
                             arma::vec& u, arma::vec& w, const arma::vec& sa2,
                             arma::mat& Phit, arma::vec& pi,
                             arma::mat& Bt, arma::vec& B2, double& sigma2,
                             double& expectedloglik, double& kldivergence,
                             int maxiter, double convtol, double epstol,
                             double stepsize, bool verbose);

// FUNCTION DEFINITIONS
// --------------------
// [[Rcpp::export]]
List flashcaisa         (const arma::mat& Y, arma::mat& A,
                         arma::vec& A2, arma::mat& B,
                         arma::vec& B2, arma::mat& PhitA, arma::vec& piA,
                         arma::mat& PhitB, arma::vec& piB,
                         const arma::vec& sa2, double& sigma2,
                         int maxiter, double epoch,
                         double convtol, double epstol,
                         double stepsize, bool verbose) {
  
  // ---------------------------------------------------------------------
  // DEFINE SIZES
  // ---------------------------------------------------------------------
  int D                  = A.n_cols;
  int K                  = sa2.n_elem;
  
  // ---------------------------------------------------------------------
  // INITIALIZE
  // ---------------------------------------------------------------------
  int iter;
  arma::vec varobj(maxiter);
  arma::vec eloglik(maxiter);
  arma::vec kldivA(maxiter);
  arma::vec kldivB(maxiter);
  arma::vec temp(D);
  arma::vec piAold(K);
  arma::vec piBold(K);
  
  // ---------------------------------------------------------------------
  // START LOOP : CYCLE THROUGH COORDINATE ASCENT UPDATES
  // ---------------------------------------------------------------------
  for (iter = 0; iter < maxiter; iter++) {
    
    piAold              = piA;
    piBold              = piB;
    
    temp                = sum(square(A), 0).t();
    updateFactor(A, Y, temp, A2, sa2, PhitB, piB, B, B2, sigma2, eloglik(iter), kldivB(iter),
                 epoch, convtol, epstol, stepsize, verbose);
    
    temp                = sum(square(B), 0).t();
    
    updateFactor(B, Y.t(), temp, B2, sa2, PhitA, piA, A, A2, sigma2, eloglik(iter), kldivA(iter),
                 epoch, convtol, epstol, stepsize, verbose);
    
    // ---------------------------------------------------------------------
    // CALCULATE VARIATIONAL OBJECTIVE
    // ---------------------------------------------------------------------
    
    varobj(iter)        = eloglik(iter) + kldivA(iter) + kldivB(iter);
    
    // ---------------------------------------------------------------------
    // CHECK CONVERGENCE
    // ---------------------------------------------------------------------
    if (max(abs(piA - piAold)) + max(abs(piB - piBold)) < convtol * 2) {
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
  
  return List::create(Named("sigma2")  = sigma2,
                      Named("kldivA")  = kldivA.subvec(0,iter-1),
                      Named("kldivB")  = kldivB.subvec(0,iter-1),
                      Named("varobj")  = varobj.subvec(0,iter-1));
}

// [[Rcpp::export]]
void updateFactor           (arma::mat& X, const arma::mat& Y,
                             arma::vec& u, arma::vec& w, const arma::vec& sa2,
                             arma::mat& Phit, arma::vec& pi,
                             arma::mat& Bt, arma::vec& B2, double& sigma2,
                             double& expectedloglik, double& kldivergence,
                             int maxiter, double convtol, double epstol,
                             double stepsize, bool verbose) {
  
  // ---------------------------------------------------------------------
  // DEFINE SIZES
  // ---------------------------------------------------------------------
  int n                  = X.n_rows;
  int p                  = X.n_cols;
  int D                  = Bt.n_rows;
  int K                  = sa2.n_elem;
  
  // ---------------------------------------------------------------------
  // INITIALIZE
  // ---------------------------------------------------------------------
  int iter               = 0;
  int j;
  arma::vec varobj(maxiter);
  arma::vec eloglik(maxiter);
  arma::vec kldiv(maxiter);
  arma::vec a(p);
  arma::vec entsum(p);
  arma::vec piold;
  
  // ---------------------------------------------------------------------
  // PRECALCULATE
  // ---------------------------------------------------------------------
  const arma::mat S1inv  = 1 / outerAddition(sa2, 1/w);
  const arma::mat S2inv  = 1 / outerAddition(1/sa2, w);
  arma::mat Rt           = Y.t() - Bt * X.t();
  arma::mat tempmat(K, p);
  
  Phit.fill(0);
  for (j = 0; j < D; j++){
    tempmat               = -S1inv / 2;
    tempmat.each_row()   %= square(Bt.row(j));
    tempmat              += log(S1inv)/2;
    tempmat.each_row()   -= max(tempmat,0);
    tempmat               = exp(tempmat);
    tempmat               = normalise(tempmat, 1, 0);
    Phit                 += tempmat / D;
  }
  pi                      = mean(Phit,1);
  
  // ---------------------------------------------------------------------
  // START LOOP : CYCLE THROUGH COORDINATE ASCENT UPDATES
  // ---------------------------------------------------------------------
  for (iter = 0; iter < maxiter; iter++) {
    
    piold              = pi;
    
    // ---------------------------------------------------------------------
    // RUN COORDINATE ASCENT UPDATES : INDEX 1 - INDEX P
    // ---------------------------------------------------------------------
    for (j = 0; j < p; j++){
      
      updateBj(X.col(j), Rt, Phit, pi, sa2, S1inv.col(j), S2inv.col(j), Bt, a, entsum,
               B2, j, w(j), u(j), sigma2, stepsize, epstol, p, D);
      
    }
    
    // ---------------------------------------------------------------------
    // CALCULATE VARIATIONAL OBJECTIVE
    // ---------------------------------------------------------------------
    eloglik(iter)         = dot(Rt,Rt)/2 - sum(square(Bt) * u)/2;
    varobj(iter)          = eloglik(iter) + sum(a)/2;
    
    sigma2                = 2 * varobj(iter) / n / D;
    
    varobj(iter)          = varobj(iter) / sigma2 + log(2*PI*sigma2)/2 * n * D -
      p * D * dot(pi, log(pi + epstol)) + sum(entsum);
    
    eloglik(iter)        += log(2*PI*sigma2)/2 * n * D + dot(B2,w) / (2 * sigma2);
    
    for (j = 1; j < K; j++){
      varobj(iter)       +=  pi(j) * log(sa2(j)) * p * D / 2 - dot( Phit.row(j), log(S2inv.row(j)) ) * D / 2;
    }
    
    kldiv(iter)           = varobj(iter) - eloglik(iter);
    
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
  expectedloglik          = eloglik(iter - 1);
  kldivergence            = kldiv(iter - 1);
  return;
  }

void updateBj              (const arma::vec& xj, arma::mat& Rt,
                            arma::mat& Phit, arma::vec& pi, const arma::vec& sa2,
                            const arma::vec& s1inv, const arma::vec& s2inv,
                            arma::mat& Bt, arma::vec& a, arma::vec& entsum,
                            arma::vec& B2, int j,
                            double wj, double uj, double& sigma2,
                            double stepsize, double epstol,
                            int p, int D) {
  
  pi                   += -Phit.col(j) / p;
  arma::vec c           = (Rt * xj + Bt.col(j) * uj) / wj;
  
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
  
  // use cpm for cpm2
  cpm                   = square(cpm); // now cpm is cpm^2
  
  // calculate a for calculating varobj = -ELBO = eloglik + kldiv
  a(j)                  = dot(sum(A % cpm,0), 1/(s2inv + epstol));
  cpm.each_row()       += s2inv.t();   // now cpm is comp2
  
  // calculate expected B square; second moment of B
  B2(j)                 = dot(cpm, A);
  
  // calculate entropy sum for kldiv
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

// [[Rcpp::export]]
void mult_eachrow(arma::mat& A, arma::vec& b) {
  A.each_row()  %= b.t();
}
