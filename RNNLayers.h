#ifndef RNNLAYERS_H
#define RNNLAYERS_H

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

typedef vector<double> Layer;
typedef vector<double> Nodes;
typedef vector<double> Frame;
typedef vector<int> Sizes;
typedef vector<vector<double> > Weight;
typedef vector<vector<double> > Data;

const int LayersNumber = 3; //total number of layers.
const int MaxVirtLayerSize = 50; // max unfolds in time




class RNNLayers {
public:

	double Weights[35][35];
	double WeightChange[35][35];
	double ContextWeight[35][35];

	Weight TransitionProb;

	Data RadialOutput;
	Data Outputlayer;
	Layer Bias;
	Layer BiasChange;
	Data Error;

	Layer Mean;
	Layer StanDev;

	Layer MeanChange;
	Layer StanDevChange;

public:
	RNNLayers();

};

#endif // RNNLAYERS_H
