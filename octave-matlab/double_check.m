x = [ 0 ; 
      0 ]
      
y = [0]

w_h = [ 0.96809199 -0.65944;
        0.92574987  0.2746975 ]

beta_h = [-0.24983311;
           0.111781  ]    
           
w_o = [0.82309361 -0.71372245]

beta_o = [ 0.17393144 ]

#Hidden Layer
z_h = w_h * x + beta_h

A_h = sigmoidFunction(z_h)

d_A_h = d_sigmoidFunction(z_h)

#Output Layer
z_o = w_o * A_h + beta_o

A_o = sigmoidFunction(z_o)

d_A_o = d_sigmoidFunction(z_o)

#Error in the output layer
delta_o = (y - A_o) .* d_A_o

#backpropagate error
delta_h = ((w_o')*delta_o) .* d_A_h

#Apply learning equations

eta = 0.1

w_o = w_o - eta*delta_o*A_h'

beta_o = beta_o - eta*delta_o

w_h = w_h - eta*delta_h*x'

beta_h = beta_h - eta*delta_h




