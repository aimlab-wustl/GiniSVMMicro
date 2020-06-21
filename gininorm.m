function [rateval,znorm] = gininorm(inpvec,gamma);
%------------------------------------------------
% This function implements the gini normalization
% procedure by finding a threshold using the
% reverse water-filling procedure.
%
% inpvec -> Unnormalized inputs.
% gamma  -> Normalization Factor.
% rateval-> Probability vector.
% znorm  -> threshold factor.
%------------------------------------------------
% Copyright (C) Shantanu Chakrabartty 2002,2012,2013,2014
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
[L,M] = size(inpvec);

rateval = zeros(1,M);
znorm = 0;

if gamma > 0,
   [fthd,ind] = sort(inpvec);
   target = gamma;
   zt = 0;
   curr = M;
   while zt < target & curr >= 2,
       zt = sum(fthd(curr:M) - fthd(curr-1));
       curr = curr - 1;
   end;
   if curr == 1,
       if zt < target,
       % Means we have to include the total classes
            znorm = 1/M*(sum(fthd(curr:M)) - target);
            for class = curr:M,
                  rateval(ind(class)) = ((inpvec(ind(class)) - znorm));
            end;
        else
            znorm = 1/(M-1)*(sum(fthd(2:M)) - target);
            for class = curr+1:M,
               rateval(ind(class)) = ((inpvec(ind(class)) - znorm));
            end;
        end;
   else
        if curr == M,
            fprintf('Ouch !!\n');
            inpvec
        end;
        znorm = 1/(M-curr)*(sum(fthd(curr+1:M)) - target);
        for class = curr+1:M,
            rateval(ind(class)) = ((inpvec(ind(class)) - znorm));
        end;
    end;
end;

