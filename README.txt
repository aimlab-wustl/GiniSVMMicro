%------------------------------------------------------------------------
% GiniSVMMicro - A Template-based Support Vector Machine
%------------------------------------------------------------------------
%
% A light-weight version of the GiniSVM multi-class probabilistic 
% support vector machine (SVM) where the user can specify:
% (1) Number of template vectors - size of the memory
% (2) Similarity function - which could be non-positive definite
%
% The script SampleScript.m contains all the documentation about how to run
% the toolkit and describes the format of the training data and the output
% produced by the model.
%
% The folder contains example synthetic data sets (mat files)
% data2class2D.mat - Linearly separable two dimensional data
% data3class2D.mat - 3 class two dimensional data
% data9class2D.mat - 9 class two dimensional data
%
% To verify the script
% 
% load data2class2D
% run SampleScript
%
% Details about the GiniSVM formulation can be found in
%
% S. Chakrabartty and G. Cauwenberghs, Gini-Support Vector Machine: Quadratic Entropy 
% Based Multi-class Probability Regression, Journal of Machine Learning Research, 
% Volume 8, pp. 813-839, April 2007.
%

