function [svout,alpha,bias,kscale] = GiniSVMTrain(trainx,Ytrain,inpB,Ntem,kscalein,svin);

%-------------------------------------------------------------------------
% This is a function to train a potential function based GiniSVM using a 
% fixed-point algorithm 
% Usage: [sv,alpha,bias,kscale] = GiniSVMTrain(trainx,Ytrain,inpB,Ntem,kscale,svin)
%
% INPUT DATA FORMAT (Required)
%-------------------------------------------------------------------------
% trainx -> input data matrix (number of data x Dimension)
% Ytrain -> input label (number of data x total classes)
% The training and cross-validation labels should prior probabilities.
% An example for a three class label is [0 0 1] to indicate
% that the training label belongs to class 3. Or it could
% be [0.1 0.3 0.6] to indicate prior confidence.
%
% INPUT PARAMETERS (Optional)
%-------------------------------------------------------------------------
% inpB = 0.5;                        % Generalization parameter
% Ntem = min(50,25 percent of data);  % Maximum support vectors allowed
% kscale = 1;                        % potential function parameter
% svin ->                            % a-priori basis vectors
%
% OUTPUT PARAMETERS
% ------------------------------------------------------------------------
% sv -> template or basis vectors
% alpha -> layer 1 weights
% bias -> layer 1 bias
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

[N,D] = size(trainx);
[Ny,M] = size(Ytrain);
if Ny ~= N,
   error('Training Data size neq labels');
end;

autopick = 0;
pickkernel = 0;
inpC = 1;
inplag = 1;
inneriter = 100000;

% Now parse the input arguments
if (nargin <2 | nargin > 6) % check correct number of arguments
    help GiniTrain
else
    if (nargin < 5) pickkernel = 1;, end
    if (nargin < 6) autopick = 1;, end
    if (nargin < 4) Ntem = min(50,floor(0.25*N));, end
    if (nargin < 3) inpB = 0.5;, end
end;


% First pick the basis vectors randomly from the balanced
% input dataset. Then pick the potential function parameter.
alpha = [];
svout = [];

curN = 0;
if (autopick ~= 1),
    svout = svin;
    [curN,Dsv] = size(svin);
    if Dsv ~= D,
       error('Input template dimension should match training dimension');
    end;
end;
Nleft = Ntem - curN;

if (Nleft > 0),
   % Pick the left over basis vectors from the training vectors
   % Create an class indicator set
   [val,indC] = max(Ytrain,[],2);
   clear val;

   NtemC = floor(Nleft/M);
    
   % Pick representative templates from training vectors
   for k = 1:M,
       dindex = find(indC == k);
       if (length(dindex) <= NtemC),
          svout = [svout; trainx(dindex,:) + ...
           0.1*(sign(rand(length(dindex),D)-0.5).*(abs(trainx(dindex,:))))];
          curN = curN + length(dindex);
       else
          rind = randperm(length(dindex));
          % Choose a template vector and then perturb it randomly
          % by 10%.
          thisvector = trainx(dindex(rind(1:NtemC)),:);
          svout = [svout; thisvector + 0.1*(sign(rand(NtemC,D)-0.5).*(abs(thisvector)))];
          curN = curN + NtemC;
       end;
   end;
end;

% Now pick the potential function parameter.
if pickkernel == 1,
   % Now choose the kernel parameters 
   done = 0;
   kscale = 0.1/D;
   nattempts = 0;
   while (done == 0) && (nattempts < 10),
      % Compute the kernel map over the template vectors
      kmap = Potential(svout,svout(1,:),kscale);
      % Estimate the average off kernel parameter
      kdiv = max(kmap(2:end))/min(kmap(2:end));
      %kdiv = kmap(1)/mean(kmap(2:end));
      if kdiv < 2*D,
         kscale = 2*kscale;
      else
         if kdiv > 4*D,
             kscale = 0.5*kscale;
         else
            done = 1;
         end;
      end;
      nattempts = nattempts + 1;
   end;
else
    kscale = kscalein;
end;

% If still more basis vectors needs to be chosen then
if Ntem - curN > 0,
   rind = randperm(N);
   svout = [svout; trainx(rind(1:Ntem-curN),:)];
end;


% Now implement the growth transform GiniSVM training
fprintf('Start GiniSVM training ...');
[alpha,bias] = GrowthSPG(trainx,Ytrain,svout,inpC,inneriter,inpB,kscale,inplag);
fprintf('...done\n');


   
