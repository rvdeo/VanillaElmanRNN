#ifndef RECURRENTNEURALNETWORK_H
#define RECURRENTNEURALNETWORK_H

//#include "RNNLayers.h"
#include "TrainingExamples.h"

#include <math.h>
#include <string>
#include <vector>
#include <algorithm>

#include <ctime>

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<ctype.h>

const double learningRate = 0.2;
const double weightdecay = 0.01;
const double MAEscalingfactor = 10;
const int maxtime = 1000; // max time in epocs


class RecurrentNeuralNetwork
{
    public:

	RNNLayers nLayer[LayersNumber];

	double Heuristic;
	Layer ChromeNeuron;
	Data Output;
	double NMSE;
	int StringSize;

	Sizes layersize;

    int row;
    int col;
    int layer;
    int r;
    int x;
    int y;


public:

	RecurrentNeuralNetwork(Sizes layer);
	RecurrentNeuralNetwork();
	double Random();

	double Sigmoid(double ForwardOutput);
	double SigmoidS(double ForwardOutput);
	double NMSError();

	void CreateNetwork(Sizes Layersize, int Maxsize);

	void ForwardPass(Samples Sample, int patternNum, Sizes Layersize,
			int phone);

	void BackwardPass(Samples Sample, double LearningRate, int slide,
			Sizes Layersize, int phone);

	void PrintWeights(Sizes Layersize); // print  all weights
	//
	bool ErrorTolerance(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	double SumSquaredError(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	int BackPropogation(TrainingExamples TraineeSamples, double LearningRate,
			Sizes Layersize, char* Savefile, char* TestFile, int sampleSize,
			int columnSize, int outputSize, ofstream &out1, ofstream &out2);

	void SaveLearnedData(Sizes Layersize, char* filename);

	void LoadSavedData(Sizes Layersize, char* filename);

	double TestLearnedData(Sizes Layersize, char* learntData, char* TestFile,
			int sampleSize, int columnSize, int outputSize);

	double CountLearningData(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	double CountTestingData(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	double MAE(TrainingExamples TraineeSamples, int temp, Sizes Layersize);

	void ChoromesToNeurons(Layer NeuronChrome);

	double ForwardFitnessPass(Layer NeuronChrome, TrainingExamples Test);

	bool CheckOutput(TrainingExamples TraineeSamples, int pattern,
			Sizes Layersize);

	double TestTrainingData(Sizes Layersize, char* learntData, char* TestFile,
			int sampleSize, int columnSize, int outputSize, ofstream & out2);

	double NormalisedMeanSquaredError(TrainingExamples TraineeSamples, int temp,
			Sizes Layersize);

	double BP(Layer NeuronChrome, TrainingExamples Test, int generations);

	double Abs(double num);

	double MAPE(TrainingExamples TraineeSamples, int temp, Sizes Layersize);

};

#endif // RECURRENTNEURALNETWORK_H
