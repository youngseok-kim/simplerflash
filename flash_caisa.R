flash_caisa = function(Y, r,
                       sa2 = (sqrt(2)^(0:7) - 1)^2, 
                       maxiter = 1000, epoch  = 2,
                       convtol = 1e-4, epstol = 1e-12,
                       stepsize = 1, outputlevel = 2,
                       verbose = FALSE) {
  
  # set size
  n            = dim(Y)[1]
  p            = dim(Y)[2]
  K            = length(sa2)
  
  # initialize by svd
  init         = propack.svd(Y, neig = r)
  r.fit        = length(init$d)
  
  # set r again
  r            = min(r, r.fit)
  
  # intialize
  A            = init$u
  B            = init$v
  mult_eachrow(A, sqrt(init$d))
  mult_eachrow(B, sqrt(init$d))
  A2           = colSums(A^2)
  B2           = colSums(B^2)
  
  PhitA        = matrix(0, K, r)
  PhitB        = matrix(0, K, r)
  piA          = double(K)
  piB          = double(K)
  
  sigma2       = mean((Y - A %*% t(B))^2);
  
  out          = flashcaisa(Y, A, A2, B, B2, PhitA, piA, PhitB, piB, sa2, sigma2,
                             maxiter, epoch, convtol, epstol, stepsize, verbose)
  iter         = length(out$varobj)
  
  if (outputlevel == 0){
    return (list(A = A, B = B, sigma2 = out$sigma2))
  } else if(outputlevel == 1){
    return (list(A = A, B = B, sigma2 = out$sigma2, varobj = out$varobj, iter = iter))
  } else if (outputlevel == 2){
    return (list(A = A, B = B, sigma2 = out$sigma2, 
                 A2 = A2, B2 = B2, piA = piA, piB = piB, PhitA = PhitA, PhitB = PhitB,
                 kldivA = out$kldivA, kldivB = out$kldivB, varobj = out$varobj, iter = iter))
  }
  return (list(out = out, A = A, B = B, A2 = A2, B2 = B2, piA = piA, piB = piB, PhitA = PhitA, PhitB = PhitB))
}