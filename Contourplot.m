function Contourplot(X,Y,trainsv,alpha,bias,inpB,kscale)

%---------------------------------------------------------------------
% 2D Contour plot to visualize the probability estimates generated
% by the machine learning algorithm
% Usage: Contourplot(X,Y,trainsv,alpha,inpB,kscale)
%
%----------------------------------------------------------------------
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

if (nargin < 7) % check correct number of arguments
    error('Usage: Contourplot(X,Y,trainsv,alpha,bias,inpB,kscale)');
end;
   
[N,M] = size(Y);
[N,D] = size(X);
[Nx,Ny]= size(trainsv);
    
% Scale the axes
xmin = min(X(:,1));, xmax = max(X(:,1)); 
ymin = min(X(:,2));, ymax = max(X(:,2)); 
    
set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);  

% Plot function value

[x,y] = meshgrid(xmin:(xmax-xmin)/25:xmax,ymin:(ymax-ymin)/25:ymax); 
for x1 = 1 : size(x,1)
   for y1 = 1 : size(x,2)
      input(1) = x(x1,y1);, input(2) = y(x1,y1);
      [prob] = GiniSVMRun(input,trainsv,alpha,bias,inpB,kscale);
      for m = 1:M,
         z(x1,y1,m) = prob(m);
      end;
      zmax(x1,y1) = max(prob);
   end
end

% Shade the contours
sp = pcolor(x,y,zmax);
shading interp;

hold on
for i = 1:size(X(:,1))
   [val,ind] = max(Y(i,:));
     
   % Multi-class class plot 
   if ind == 1,
      plot(X(i,1),X(i,2),'wo','LineWidth',1) % Class A
   else
      if ind == 2,
           plot(X(i,1),X(i,2),'w+','LineWidth',1) % Class B
      else
         if ind == 3,
              plot(X(i,1),X(i,2),'w*','LineWidth',1) % Class B
         else
             if ind == 4,
                 plot(X(i,1),X(i,2),'ws','LineWidth',1) % Class B
             else
                 plot(X(i,1),X(i,2),'w^','LineWidth',1) % Class B
             end;
         end;
      end;
   end;
end;

% Plot Boundary contour

hold on
for m = 1:M,
    contour(x,y,z(:,:,m),[0.5 0.5],'w--')
end;
xlabel('x1');
ylabel('x2');
hold off
colormap(gray);
axis square;
  
