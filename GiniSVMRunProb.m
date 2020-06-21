function [layer1out] = GiniSVMRunProb(x,trainsv,alpha,bias,inpB,kscale);
%------------------------------------------------
% This function evaluates the trained GiniSVM producing 
% an M dimensional conditional probability estimate.
%
% x -> input.
% trainsv  -> template vector array.
% alpha -> weight vector array.
% inpB  -> generalization factor.
% kscale -> potential function parameter
% layer1out -> output probability vector
%------------------------------------------------------------------------
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

% Sanity check to ensure all dimensions match.
[N,D] = size(x);
[Ndata,Nw] = size(trainsv);
[Nalpha,M] = size(alpha);
if Nw~=D
    fprintf('Error: W and x different number of data points (%g/%g)\n\n', Nw, D);
    return
end;

layer1out = zeros(N,M);

for k = 1:N,
   mvalue = Potential(x(k,:),trainsv,kscale)*alpha + bias;
   [result, resultmargin] = gininorm(mvalue,2*inpB);
   layer1out(k,:) = result/(2*inpB);
end;
    

