function [W,b] = GrowthSPG(x, y, z, C, numofiter, B, kscale, lag);
%-------------------------------------------------------------------------
% function [W, b] = GrowthSPG(x, y, inpC, numofiter, inpB,kscale);
%
%      Potential function GiniSVM regression using fixed-point algorithm.
%
%      x: independent variable, (L,N) with L: number of points; N: dimension
%      y: dependent variable, (L,M) containing class labels vector
%      z: sample template vectors
%      C: regularization constant is a vector that is obtained
%         by weighing the results from the previous decisions
%      numberofiter: Total number of iterations allowed.
%      numofiter : Total number of optimization steps.
%      B: Generalization Factor.
%      lag: relaxation parameter for bias computation.
%
%      W: weight vector (L,M)
%      b: bias vector (1,M)
%-------------------------------------------------------------------------
% Copyright (C) Shantanu Chakrabartty 2002,2012,2013,2014,2015
% Version: GiniSVMMicrov1.0
%-------------------------------------------------------------------------
% Licensing Terms: This program is granted free of charge for research and 
% education purposes. However you must obtain a license from the author to 
% use it for commercial purposes. The software must not be modified and 
% distributed without prior permission of the author. By using this 
% software you agree to the licensing terms:
%
% NO WARRANTY: BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO 
% WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. 
% EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR 
% OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, 
% EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE 
% ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.
% SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY 
% SERVICING, REPAIR OR CORRECTION. IN NO EVENT UNLESS REQUIRED BY 
% APPLICABLE LAW OR AGREED TO IN WRITING WILL ANY COPYRIGHT HOLDER, OR ANY 
% OTHER PARTY WHO MAY MODIFY AND/OR REDISTRIBUTE THE PROGRAM, BE LIABLE TO 
% YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR 
% CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE 
% PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING 
% RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A 
% FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH 
% HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH 
% DAMAGES. 
%-------------------------------------------------------------------------


%-----------------------------------------------------------------------------------
% First compute the size of the training data
%-----------------------------------------------------------------------------------
[L,N] = size(x);
[Ly,M] = size(y);
[Lz,D] = size(z);
if Ly~=L
    fprintf('Error: x and y different number of data points (%g/%g)\n\n', L, Ly);
    return
end;
              

%-----------------------------------------------------------------------------------
% First initialize all the variables
%-----------------------------------------------------------------------------------

% Precompute the inner kernel with respect to the templates
Q = Potential(x,z,kscale);

%-----------------------------------------------------------------------------------
% Intialize iteration variables
%-----------------------------------------------------------------------------------
converged = 0;

% Target cost reduction of 0.01%
tol = 0.1;

% Convergence hyperparameter
Cs = 1;
Fac = 2*(2*Cs+B/C)*(L + 1);

iter = 1;

P = 1/M*ones(L,M);
W = Q'*(y-P); 

% Estimate the bias
b = sum(y-P,1);

% Evaluate the current objective function
Costp = 0;
for class1 = 1:M, 
    Costp = Costp + 0.5*W(:,class1)'*W(:,class1) + 0.5*lag*b*b';
    Costp = Costp + 0.5*sum(B/C*((P(:,class1)).^2 ));
end; 

while converged == 0,  
   
   G = Q*W + ones(L,1)*b*lag - (B/C)*P;
   P = P.*(G + Fac);
   P = P./(sum(P,2)*ones(1,M));
   
   % New weights
   W = Q'*(y-P);
   b = sum(y-P,1);

           
   if rem(iter,1000) == 0,
       
      % Evaluate the current objective function
      Cost = 0;
      for class1 = 1:M, 
         Cost = Cost + 0.5*W(:,class1)'*W(:,class1) + 0.5*lag*b*b';
         Cost = Cost + 0.5*sum(B/C*((P(:,class1)).^2 ));
      end;
   
      errval = (Costp - Cost)/Costp * 100;
      fprintf('\n Cost and decrease = %d, %d percent',Cost,errval);
      
      Costp = Cost;
      
      if errval < 0,
         fprintf('\n Cost function increased !!! .. reseting and restarting...\n');
         P = 1/M*ones(L,M);
         W = Q'*(y-P);  
         Cs = 2*Cs;
         Fac = 2*(2*Cs+B/C)*(L + 1);
         b = sum(y-P,1);
       
         % Evaluate the current objective function
         Costp = 0;
         for class1 = 1:M, 
            Costp = Costp + 0.5*W(:,class1)'*W(:,class1) + 0.5*lag*b*b';
            Costp = Costp + 0.5*sum(B/C*((P(:,class1)).^2 ));
         end;  
      elseif (errval < tol),
         converged = 1;
      end;      
   end;
   
   if (iter > numofiter), 
       converged = 1;
   end;
     
   iter = iter+1;
end;

% Update the bias
b = lag*b;
