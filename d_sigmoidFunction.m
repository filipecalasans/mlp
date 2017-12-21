function g = d_sigmoidFunction(z)
   g = sigmoidFunction(z) .* (1 + sigmoidFunction(z))
endfunction