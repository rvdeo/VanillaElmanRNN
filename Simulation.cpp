#include "Simulation.h"

Simulation::Simulation() {
    ;
}

int Simulation::GetEval() {
    return TotalEval;
}
double Simulation::GetCycle() {
    return Train;
}
double Simulation::GetError() {
    return Test;
}

double Simulation::NMSETrain() {
    return TrainNMSE;
}
double Simulation::NMSETest() {
    return TestNMSE;
}

bool Simulation::GetSucess() {
    return Sucess;
}


void Simulation::Procedure(bool bp, double h, ofstream &out1, ofstream &out2,
		ofstream &out3) {

	clock_t start = clock();

	int hidden = h;

	int output = 1;
	int input = 1;

	int weightsize1 = (input * hidden);

	int weightsize2 = (hidden * output);

	int contextsize = hidden * hidden;
	int biasize = hidden + output;

	int gene = 1;
	double trainpercent = 0;
	double testpercent = 0;
	int epoch;
	double testtree;

	char file[15] = "Learnt.txt";
	TotalEval = 0;
	double H = 0;

	TrainingExamples Samples(trainfile, trainsize, input, output);
	 Samples.printData();

	double error;

	Sizes layersize;
	layersize.push_back(input);
	layersize.push_back(hidden);
	layersize.push_back(output);

	RecurrentNeuralNetwork network(layersize);
	network.CreateNetwork(layersize, trainsize);

	epoch = network.BackPropogation(Samples, learningRate, layersize, file,
			trainfile, trainsize, input, output, out1, out2); //  train the network

	out2 << "Train" << endl;
	Train = network.TestTrainingData(layersize, file, trainfile, trainsize,
			input, output, out2);
	TrainNMSE = network.NMSError();
	out2 << "Test" << endl;
	Test = network.TestTrainingData(layersize, file, testfile, testsize, input,
			output, out2);
	TestNMSE = network.NMSError();
	out2 << endl;
	cout << Test << " was test RMSE " << endl;
	out1 << endl;
	out1 << " ------------------------------ " << h << "  " << TotalEval
			<< "  RMSE:  " << Train << "  " << Test << " NMSE:  " << TrainNMSE
			<< " " << TestNMSE << endl;

	out2 << " ------------------------------ " << h << "  " << TotalEval << "  "
			<< Train << "  " << Test << endl;
	out3 << "  " << h << "  " << TotalEval << "  RMSE:  " << Train << "  "
			<< Test << " NMSE:  " << TrainNMSE << " " << TestNMSE << endl;

}
