%-------------------------------------------------------------------------
% Test script to show how to train and run GiniSVM
%-------------------------------------------------------------------------
% Uses the following data for training and evaluation
% trainx -> input data matrix (number of data x Dimension)
% Ytrain -> input label (number of data x total classes)
% crossx -> cross-validation data (number of data x Dimension)
% Ycross -> cross-validation label (number of data x Dimension)
% The training and cross-validation labels should be prior probabilities.
% An example for a three class label is [0 0 1] to indicate
% that the training label belongs to class 3. Or it could
% be [0.1 0.3 0.6] to indicate prior confidence.
%
% There are two parameters that users can use to control the
% properties of this machine.
%
% inpB -> This parameter controls the generalization property of the
% the learner. If a smaller value of inpB is chosen, it will try to
% overfit to the training data. Note that choosing a smaller inpB increases 
% the training time.
%
% Ntem -> This specifies the number of basis vectors allowed. This
% parameter controls how accurately you want to capture different
% parts of the input space. Smaller Ntem means you want to coarsely
% capture the probability contours.
% 
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


%-------------------------------------------------------------------------
% THESE ARE THE PARAMETERS USERS SHOULD OPTIMIZE OR MODIFY
%-------------------------------------------------------------------------
inpB = 0.1;    % Generalization parameter.
kscale = 0.2;
Ntem = 10;   % number of basis vectors.
%-------------------------------------------------------------------------



% Starting from this point only modify if you know what your application
% requires

% First get all the dimensions of the input data
[N,D] = size(trainx);
[Ny,M] = size(Ytrain);
if Ny ~= N,
   error('Training Data size neq labels');
end;

[Ncross,D] = size(crossx);
[Nycross,M] = size(Ycross);
if Nycross ~= Ncross,
   error('Cross-validation Data size neq labels');
end;


%--------------------------------
% Flag parameters
%--------------------------------
plotflag = 1;
%--------------------------------


for i = 1:N,
   [val,yindex(i)] = max(Ytrain(i,:) + 1e-6*rand(1,M));
end;

for i = 1:Ncross,
   [val,ycrossindex(i)] = max(Ycross(i,:) + 1e-6*rand(1,M));
end;

% -----------------------------------------------------------------
% Here are the three variants of the Training function that 
% can be used
% -----------------------------------------------------------------

% Here is an example of training when the basis vectors
% are not specified and the potential function parameter
% is not specified.
[sv,alpha,bias,kscale] = GiniSVMTrain(trainx,Ytrain,inpB,Ntem,kscale);

% Here is an example of training when the basis vectors are
% not specified but the potential function parameter is specified. 
%[sv,alpha,bias] = GiniSVMTrain(trainx,Ytrain,inpB,Ntem,kscale);

% Here is an example when the basis vectors and the potential
% function parameter are specified. 
%[sv,alpha,bias] = GiniSVMTrain(trainx,Ytrain,inpB,Ntem,kscale,svin);

   
%------------------------------------------------------------------
% Below is the script to evaluate the performance on training set
%------------------------------------------------------------------

fprintf('Evaluating Performance on Training set\n');
errordist = zeros(1,M);
error = 0;
etotal = zeros(1,M);
confusionTrain = zeros(M,M);

[result] = GiniSVMRunRaw(trainx,sv,alpha,bias,inpB,kscale);
for k = 1:N,
   [maxval,ind] = max(result(k,:));
   etotal(yindex(k)) = etotal(yindex(k)) + 1;
   confusionTrain(ind,yindex(k)) = confusionTrain(ind,yindex(k)) + 1;
   if ind ~= yindex(k),
      error = error + 1;
      errordist(yindex(k)) = errordist(yindex(k)) + 1;
   end;
end;
fprintf('Multi-class Train Error = %d percent \n',ceil((error/N)*100));
for i = 1:M,
   fprintf('Class %d: Total Data = %d; Error = %d percent \n',i,etotal(i),floor((errordist(i)/(etotal(i)+1e-9))*100));
end;
trainresult = result;
clear result resultmargin;

% Now print the confusion matrix after normalization
confusionTrain = confusionTrain./(ones(M,1)*(etotal+1e-9))*100;
fprintf('Training Confusion Matrix\n');
floor(confusionTrain)
        

%------------------------------------------------------------------
% Below is the script to evaluate the performance on validation set
%------------------------------------------------------------------
fprintf('Evaluating Performance on Cross-validation set\n');
errordist = zeros(1,M);
error = 0;
etotalcross = zeros(1,M);
confusionCross = zeros(M,M);

[result] = GiniSVMRunRaw(crossx,sv,alpha,bias,inpB,kscale);
for k = 1:Ncross,
   [maxval,ind] = max(result(k,:));
   etotalcross(ycrossindex(k)) = etotalcross(ycrossindex(k)) + 1;
   confusionCross(ind,ycrossindex(k)) = confusionCross(ind,ycrossindex(k)) + 1;
   if ind ~= ycrossindex(k),
      error = error + 1;
      errordist(ycrossindex(k)) = errordist(ycrossindex(k)) + 1;
   end;
end;
fprintf('Multi-class Cross-validation Error = %d percent \n',ceil((error/Ncross)*100));
for i = 1:M,
   fprintf('Class %d: Total Data = %d; Error = %d percent \n',i,etotalcross(i),floor((errordist(i)/(etotalcross(i)+1e-9))*100));
end;
crossresult = result;
clear result resultmargin;

% Now print the confusion matrix after normalization
confusionCross = confusionCross./(ones(M,1)*(etotalcross+1e-9))*100;
fprintf('Cross-validation Confusion Matrix\n');
floor(confusionCross)

%---------------------------------------
% Plot the probability Contour for 2D data
%---------------------------------------
if ((plotflag == 1) & (D == 2)),
   fprintf('Plotting Contour ....');
   figure;
   Contourplot(trainx,Ytrain,sv,alpha,bias,inpB,kscale);
   fprintf('....done\n');
end;
