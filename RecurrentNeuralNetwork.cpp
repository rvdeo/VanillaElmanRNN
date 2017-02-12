#include "RecurrentNeuralNetwork.h"

RecurrentNeuralNetwork::RecurrentNeuralNetwork() {
    ;
}

RecurrentNeuralNetwork::RecurrentNeuralNetwork(Sizes layer) {
		layersize = layer;

		StringSize = (layer[0] * layer[1]) + (layer[1] * layer[2])
				+ (layer[1] * layer[1]) + (layer[1] + layer[2]);

	}

double RecurrentNeuralNetwork::NMSError() {
		return NMSE;
}

double RecurrentNeuralNetwork::Random() {
	int chance;
	double randomWeight = 0;
	double NegativeWeight = 0;
	chance = rand() % 2;

	if (chance == 0) {

		return drand48() * 0.5;
	}

	if (chance == 1) {

		return drand48() * 0.5;
	}

}

double RecurrentNeuralNetwork::Sigmoid(double ForwardOutput) {
	double ActualOutput;

	ActualOutput = (1.0 / (1.0 + exp(-1.0 * (ForwardOutput))));

	return ActualOutput;
}

double RecurrentNeuralNetwork::SigmoidS(double ForwardOutput) {
	double ActualOutput;

	ActualOutput = (1.0 / (1.0 + exp(-1.0 * (ForwardOutput))));

	return ActualOutput;
}

double RecurrentNeuralNetwork::Abs(double num) {
	if (num < 0)
		return num * -1;
	else
		return num;
}

void RecurrentNeuralNetwork::CreateNetwork(Sizes Layersize, int Maxsize) {

	int end = Layersize.size() - 1;

	for (layer = 0; layer < Layersize.size() - 1; layer++) {

		//-------------------------------------------
		//for( r=0; r < Layersize[layer]; r++)
		//nLayer[layer].Weights.push_back(vector<double> ());

		for (row = 0; row < Layersize[layer]; row++)
			for (col = 0; col < Layersize[layer + 1]; col++)
				nLayer[layer].Weights[row][col] = Random();
		//---------------------------------------------

		for (row = 0; row < Layersize[layer]; row++)
			for (col = 0; col < Layersize[layer + 1]; col++)
				nLayer[layer].WeightChange[row][col] = Random();
		//-------------------------------------------
	}

	for (row = 0; row < Layersize[1]; row++)
		for (col = 0; col < Layersize[1]; col++)
			nLayer[1].ContextWeight[row][col] = Random();

	//}
	//------------------------------------------------------

	for ( layer = 0; layer < Layersize.size(); layer++) {

		for (r = 0; r < Maxsize; r++)
			nLayer[layer].Outputlayer.push_back(vector<double>());

		for (row = 0; row < Maxsize; row++)
			for (col = 0; col < Layersize[layer]; col++)
				nLayer[layer].Outputlayer[row].push_back(Random());
		//--------------------------------------------------

		for (r = 0; r < MaxVirtLayerSize; r++)
			nLayer[layer].Error.push_back(vector<double>());

		for (row = 0; row < MaxVirtLayerSize; row++)
			for (col = 0; col < Layersize[layer]; col++)
				nLayer[layer].Error[row].push_back(0);

		//TransitionProb
		//---------------------------------------------
		for (r = 0; r < MaxVirtLayerSize; r++)
			nLayer[layer].RadialOutput.push_back(vector<double>());

		for (row = 0; row < MaxVirtLayerSize; row++)
			for (col = 0; col < Layersize[layer]; col++)
				nLayer[layer].RadialOutput[row].push_back(Random());
		//-------------------------------------------

		//---------------------------------------------

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Bias.push_back(Random());

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].BiasChange.push_back(0);

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].Mean.push_back(Random());

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].StanDev.push_back(Random());

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].MeanChange.push_back(0);

		for (row = 0; row < Layersize[layer]; row++)
			nLayer[layer].StanDevChange.push_back(0);

	}
	//--------------------------------------

	for (r = 0; r < Maxsize; r++)
		Output.push_back(vector<double>());
	for (row = 0; row < Maxsize; row++)
		for (col = 0; col < Layersize[end]; col++)
			Output[row].push_back(0);

	for (row = 0; row < StringSize; row++)
		ChromeNeuron.push_back(0);

	// SaveLearnedData(Layersize, "createnetwork.txt");

}

void RecurrentNeuralNetwork::ForwardPass(Samples Sample, int slide, Sizes Layersize,
		int phone) {
	double WeightedSum = 0;
	double ContextWeightSum = 0;
	double ForwardOutput;
	//  cout<<endl<<"slide  "<<slide<<"  ------------------------ "<<endl<<endl<<endl<<endl;

	int end = Layersize.size() - 1;

	for (int row = 0; row < Layersize[0]; row++)
		nLayer[0].Outputlayer[slide + 1][row] = Sample.InputValues[slide][row];
	//--------------------------------------------

	//for(
	int layer = 0; // layer < Layersize.size()-1; layer++){
	int y;
	int x;
	for (y = 0; y < Layersize[layer + 1]; y++) {
		for (x = 0; x < Layersize[layer]; x++) {
			WeightedSum += (nLayer[layer].Outputlayer[slide + 1][x]
					* nLayer[layer].Weights[x][y]);
		}
		for (x = 0; x < Layersize[layer + 1]; x++) {
			ContextWeightSum += (nLayer[1].Outputlayer[slide][x]
					* nLayer[1].ContextWeight[x][y]); // adjust this line when use two hidden layers.
			//
		}

		ForwardOutput = (WeightedSum + ContextWeightSum)
				- nLayer[layer + 1].Bias[y];
		nLayer[layer + 1].Outputlayer[slide + 1][y] = SigmoidS(ForwardOutput);
		// cout<<ForwardOutput<<endl;
		//getchar();
		WeightedSum = 0;
		ContextWeightSum = 0;
	}
//}//end layer

	layer = 1;

	for (y = 0; y < Layersize[layer + 1]; y++) {
		for (x = 0; x < Layersize[layer]; x++) {
			WeightedSum += (nLayer[layer].Outputlayer[slide + 1][x]
					* nLayer[layer].Weights[x][y]);
			ForwardOutput = (WeightedSum) - nLayer[layer + 1].Bias[y];
		}
		nLayer[layer + 1].Outputlayer[slide + 1][y] = Sigmoid(ForwardOutput);
		WeightedSum = 0;
		//cout<<   ForwardOutput<<endl;
		ContextWeightSum = 0;
	}

	//--------------------------------------------
	for (int output = 0; output < Layersize[end]; output++) {
		Output[phone][output] = nLayer[end].Outputlayer[slide + 1][output];
//cout<<Output[phone][output]<<" ";
	}
//cout<<endl;

}

void RecurrentNeuralNetwork::BackwardPass(Samples Sample, double LearningRate, int slide,
		Sizes Layersize, int phone) {

	int end = Layersize.size() - 1; // know the end layer
	double temp = 0;
	double sum = 0;
	int Endslide = Sample.PhoneSize;
	//----------------------------------------
//cout<<slide<<"   ---------------------------------->>>>"<<endl;
	// compute error gradient for output neurons
	for (int output = 0; output < Layersize[end]; output++) {
		nLayer[2].Error[Endslide][output] = 1
				* (Sample.OutputValues[output] - Output[phone][output]);

	}
//

	//----------------------------------------
	// for(int layer = Layersize.size()-2; layer >= 0; layer--){
	int layer = 1;
	for (x = 0; x < Layersize[layer]; x++) { //inner layer
		for (y = 0; y < Layersize[layer + 1]; y++) { //outer layer
			temp += (nLayer[layer + 1].Error[Endslide][y]
					* nLayer[layer].Weights[x][y]);
		}

		nLayer[layer].Error[Endslide][x] =
				nLayer[layer].Outputlayer[Endslide][x]
						* (1 - nLayer[layer].Outputlayer[Endslide][x]) * temp;
		temp = 0.0;
	}
	// }
//cout<<nLayer[1].Error[Endslide][0]<<"        eeee"<<endl;
	for (x = 0; x < Layersize[1]; x++) { //inner layer
		for (y = 0; y < Layersize[1]; y++) { //outer layer
			sum += (nLayer[1].Error[slide][y] * nLayer[1].ContextWeight[x][y]);
//cout<<sum<< " is sum  : "<<nLayer[1].Error[slide][y]<<endl;

		}
		nLayer[1].Error[slide - 1][x] = (nLayer[1].Outputlayer[slide - 1][x]
				* (1 - nLayer[1].Outputlayer[slide - 1][x])) * sum;
		//	cout<<	nLayer[1].Error[slide-1][x]<<" is error  of slide "<<slide<<endl;
		sum = 0.0;
	}
	sum = 0.0;

// do weight updates..
//---------------------------------------
	double tmp;
//for( layer = Layersize.size()-2; layer != -1; layer--){
	for (x = 0; x < Layersize[0]; x++) { //inner layer
		for (y = 0; y < Layersize[1]; y++) { //outer layer
			tmp = ((LearningRate * nLayer[1].Error[slide][y]
					* nLayer[0].Outputlayer[slide][x])); // weight change
			nLayer[0].Weights[x][y] += tmp - (tmp * weightdecay);
		}

	}
	// }

//cout<<endl;

//-------------------------------------------------
//do top weight update
	double seeda = 0;

	//if(Endslide ==  slide)
	seeda = 1;
	double tmpoo;
//for( layer = Layersize.size()-2; layer != -1; layer--){
	for (x = 0; x < Layersize[1]; x++) { //inner layer
		for (y = 0; y < Layersize[2]; y++) { //outer layer
			tmpoo = ((seeda * LearningRate * nLayer[2].Error[Endslide][y]
					* nLayer[1].Outputlayer[Endslide][x])); // weight change
			nLayer[1].Weights[x][y] += tmpoo - (tmpoo * weightdecay);

		}

	}
	seeda = 0;
	// }

	//-----------------------------------------------
	double tmp2;
//for( layer = Layersize.size()-2; layer != -1; layer--){
	for (x = 0; x < Layersize[1]; x++) { //inner layer
		for (y = 0; y < Layersize[1]; y++) { //outer layer
			tmp2 = ((LearningRate * nLayer[1].Error[slide][y]
					* nLayer[1].Outputlayer[slide - 1][x])); // weight change
			nLayer[1].ContextWeight[x][y] += tmp2 - (tmp2 * weightdecay);

		}

	}

//update the bias
	double topbias = 0;
	double seed = 0;

	//if(Endslide ==  slide)
	seed = 1;
	for (y = 0; y < Layersize[2]; y++) {
		topbias = ((seed * -1 * LearningRate * nLayer[2].Error[Endslide][y]));
		nLayer[2].Bias[y] += topbias - (topbias * weightdecay);
		topbias = 0;
		//   	cout<<nLayer[2].Bias[y]<<" is updated  top Bias for slide "<<Endslide<<endl;
	}
	topbias = 0;
	seed = 0;

	double tmp1;

	for (y = 0; y < Layersize[1]; y++) {
		tmp1 = ((-1 * LearningRate * nLayer[1].Error[slide][y]));
		nLayer[1].Bias[y] += tmp1 - (tmp1 * weightdecay);

	}
	// }

}

double RecurrentNeuralNetwork::MAPE(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	for (int pattern = 0; pattern < temp; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = (TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output])
					/ TraineeSamples.Sample[pattern].OutputValues[output];

			ErrorSquared += fabs(Error);
		}

		Sum += (ErrorSquared);
		ErrorSquared = 0;
	}
	return (Sum / temp * Layersize[end] * 100);

}

double RecurrentNeuralNetwork::MAE(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	for (int pattern = 0; pattern < temp; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = (TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output]) * MAEscalingfactor;

			ErrorSquared += fabs(Error);
		}

		Sum += (ErrorSquared);
		ErrorSquared = 0;
	}
	return Sum / temp * Layersize[end];

}
double RecurrentNeuralNetwork::SumSquaredError(TrainingExamples TraineeSamples, int temp,
		Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Error = 0;
	double ErrorSquared = 0;
	for (int pattern = 0; pattern < temp; pattern++) {
		for (int output = 0; output < Layersize[end]; output++) {
			Error = TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output];

			ErrorSquared += (Error * Error);
		}

		Sum += (ErrorSquared);
		ErrorSquared = 0;
	}
	return sqrt(Sum / temp * Layersize[end]);

//return MAPE(TraineeSamples,temp,Layersize);
}

double RecurrentNeuralNetwork::NormalisedMeanSquaredError(
		TrainingExamples TraineeSamples, int temp, Sizes Layersize) {
	int end = Layersize.size() - 1;
	double Sum = 0;
	double Sum2 = 0;
	double Error = 0;
	double ErrorSquared = 0;
	double Error2 = 0;
	double ErrorSquared2 = 0;
	double meany = 0;
	for (int pattern = 0; pattern < temp; pattern++) {

		for (int slide = 0; slide < TraineeSamples.Sample[pattern].PhoneSize;
				slide++) {
			for (int input = 0; input < Layersize[0]; input++) {
				meany +=
						TraineeSamples.Sample[pattern].InputValues[slide][input];
			}
			meany /= Layersize[0] * TraineeSamples.Sample[pattern].PhoneSize;
		}

		for (int output = 0; output < Layersize[end]; output++) {
			Error2 = TraineeSamples.Sample[pattern].OutputValues[output]
					- meany;
			Error = TraineeSamples.Sample[pattern].OutputValues[output]
					- Output[pattern][output];
			ErrorSquared += (Error * Error);
			ErrorSquared2 += (Error2 * Error2);

		}
		meany = 0;
		Sum += (ErrorSquared);
		Sum2 += (ErrorSquared2);
		ErrorSquared = 0;
		ErrorSquared2 = 0;
	}

	return Sum / Sum2;
}

void RecurrentNeuralNetwork::PrintWeights(Sizes Layersize) {
	int end = Layersize.size() - 1;

	for (int layer = 0; layer < Layersize.size() - 1; layer++) {

		cout << layer << "  Weights::" << endl << endl;
		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++)
				cout << nLayer[layer].Weights[row][col] << " ";
			cout << endl;
		}
		cout << endl << layer << " ContextWeight::" << endl << endl;

		for (int row = 0; row < Layersize[1]; row++) {
			for (int col = 0; col < Layersize[1]; col++)
				cout << nLayer[1].ContextWeight[row][col] << " ";
			cout << endl;
		}

	}

}
//-------------------------------------------------------

void RecurrentNeuralNetwork::SaveLearnedData(Sizes Layersize, char* filename) {

	ofstream out;
	out.open(filename);
	if (!out) {
		cout << endl << "failed to save file" << endl;
		return;
	}
	//-------------------------------
	for (int layer = 0; layer < Layersize.size() - 1; layer++) {
		for (int row = 0; row < Layersize[layer]; row++) {
			for (int col = 0; col < Layersize[layer + 1]; col++)
				out << nLayer[layer].Weights[row][col] << " ";
			out << endl;
		}
		out << endl;
	}
	//-------------------------------
	for (int row = 0; row < Layersize[1]; row++) {
		for (int col = 0; col < Layersize[1]; col++)
			out << nLayer[1].ContextWeight[row][col] << " ";
		out << endl;
	}
	out << endl;
	//--------------------------------
	// output bias.
	for (int layer = 1; layer < Layersize.size(); layer++) {
		for (int y = 0; y < Layersize[layer]; y++) {
			out << nLayer[layer].Bias[y] << "  ";
			out << endl;
		}

	}
	out << endl;
	//------------------------------
	out.close();
//	cout << endl << "data saved" << endl;

	return;
}

void RecurrentNeuralNetwork::LoadSavedData(Sizes Layersize, char* filename) {
	ifstream in(filename);
	if (!in) {
		cout << endl << "failed to load file" << endl;
		return;
	}
	//-------------------------------
	for (int layer = 0; layer < Layersize.size() - 1; layer++)
		for (int row = 0; row < Layersize[layer]; row++)
			for (int col = 0; col < Layersize[layer + 1]; col++)
				in >> nLayer[layer].Weights[row][col];
	//---------------------------------
	for (int row = 0; row < Layersize[1]; row++)
		for (int col = 0; col < Layersize[1]; col++)
			in >> nLayer[1].ContextWeight[row][col];
	//--------------------------------
	// output bias.
	for (int layer = 1; layer < Layersize.size(); layer++)
		for (int y = 0; y < Layersize[layer]; y++)
			in >> nLayer[layer].Bias[y];

	in.close();
	// cout << endl << "data loaded for testing" << endl;

	return;
}

double RecurrentNeuralNetwork::TestTrainingData(Sizes Layersize, char* learntData,
		char* TestFile, int sampleSize, int columnSize, int outputSize,
		ofstream & out2) {
	bool valid;
	double count = 1;
	double total;
	double accuracy;
	int end = Layersize.size() - 1;
	Samples sample;
	TrainingExamples Test(TestFile, sampleSize, columnSize, outputSize);

	for (int phone = 0; phone < Test.SampleSize; phone++) {
		sample = Test.Sample[phone];

		int slide;

		for (slide = 0; slide < sample.PhoneSize; slide++) {
			ForwardPass(sample, slide, Layersize, phone);

		}
	}

	// for(int pattern = 0; pattern< Test.SampleSize; pattern++){     //in case if you wish to print the prediction outputs
	//out2<< Output[pattern][0]  <<" "<<Test.Sample[pattern].OutputValues[0] <<endl;

	//  }
//out2<<endl;

	out2 << endl;
	accuracy = SumSquaredError(Test, Test.SampleSize, Layersize);
	out2 << " RMSE:  " << accuracy << endl;
	cout << "RMSE: " << accuracy << " %" << endl;
	NMSE = MAE(Test, Test.SampleSize, Layersize);
	out2 << " NMSE:  " << NMSE << endl;
	return accuracy;
}

int RecurrentNeuralNetwork::BackPropogation(TrainingExamples TraineeSamples,
		double LearningRate, Sizes Layersize, char * Savefile, char* TestFile,
		int sampleSize, int columnSize, int outputSize, ofstream &out1,
		ofstream &out2) {

	double SumErrorSquared;

	Sizes Array;

	Samples sample;

	CreateNetwork(Layersize, TraineeSamples.SampleSize);

	int Id = 0;

	int c = 1;

	int num ;
	bool sockhastic = true;

	for (int epoch = 0; epoch < maxtime; epoch++) {

		//TraineeSamples.SampleSize
		for (int phone = 0; phone < TraineeSamples.SampleSize; phone++) {

            if(sockhastic ==true)
                num = rand()%TraineeSamples.SampleSize;
            else
                num = phone;

            sample = TraineeSamples.Sample[num];

			int slide;

			for (slide = 0; slide < sample.PhoneSize; slide++) {

				ForwardPass(sample, slide, Layersize, num);

			}

			for (slide = sample.PhoneSize; slide >= 1; slide--) {
				BackwardPass(sample, LearningRate, slide, Layersize, num);
			}

		}

		double Train = 0;
		if (epoch % 100 == 0) {
			SumErrorSquared = SumSquaredError(TraineeSamples,
					TraineeSamples.SampleSize, Layersize);

			double mae = MAE(TraineeSamples, TraineeSamples.SampleSize,
					Layersize);
			cout << SumErrorSquared << "     " << mae << " " << epoch << endl;

			out1 << SumErrorSquared << "     " << mae << " " << " " << Train
					<< epoch << endl;
		}

	}

	SaveLearnedData(Layersize, Savefile);

	return c;

}

