#ifndef SIMULATION_H
#define SIMULATION_H

#include "RecurrentNeuralNetwork.h"


const double MinimumError = 0.00001;



class Simulation {

public:
	int TotalEval;
	int TotalSize;
	double Train;
	double Test;
	double TrainNMSE;
	double TestNMSE;
	double Error;

	const int trainsize = 299;
    const int testsize = 99;

    char * trainfile = "train_embed.txt"; // time series after state state reconstruction (time lag = 2, dimen = 3)
    char * testfile = "test_embed.txt"; //


	int Cycles;
	bool Sucess;

	Simulation();
	int GetEval() ;
	double GetCycle() ;
	double GetError() ;
	double NMSETrain() ;
	double NMSETest() ;
	bool GetSucess() ;

	void Procedure(bool bp, double h, ofstream &out1, ofstream &out2,
			ofstream &out3);
};


#endif // SIMULATION_H
