#include "TrainingExamples.h"


TrainingExamples::TrainingExamples(){
    ;
}

TrainingExamples::TrainingExamples(char* File, int sampleSize, int columnSize,
			int outputSize) {

		Samples sample;

		for (int i = 0; i < sampleSize; i++) {
			Sample.push_back(sample);
		}

		int rows;
		RowSize = MaxVirtLayerSize; // max number of rows.
		ColumnSize = columnSize;
		SampleSize = sampleSize;
		OutputSize = outputSize;

		ifstream in(File);

		//initialise input vectors
		for (int sample = 0; sample < SampleSize; sample++) {

			for (int r = 0; r < RowSize; r++)
				Sample[sample].InputValues.push_back(vector<double>());

			for (int row = 0; row < RowSize; row++) {
				for (int col = 0; col < ColumnSize; col++)
					Sample[sample].InputValues[row].push_back(0);
			}

			for (int out = 0; out < OutputSize; out++)
				Sample[sample].OutputValues.push_back(0);
		}
		//---------------------------------------------

		for (int samp = 0; samp < SampleSize; samp++) {
			in >> rows;
			Sample[samp].PhoneSize = rows;

			for (row = 0; row < Sample[samp].PhoneSize; row++) {
				for (col = 0; col < ColumnSize; col++)
					in >> Sample[samp].InputValues[row][col];
			}

			for (int out = 0; out < OutputSize; out++)
				in >> Sample[samp].OutputValues[out];

			// cout<<rows<<endl;
		}

		cout << "printing..." << endl;

		in.close();
}

void TrainingExamples::printData() {
	for (int sample = 0; sample < SampleSize; sample++) {
		for (row = 0; row < Sample[sample].PhoneSize; row++) {
			for (col = 0; col < ColumnSize; col++)
				cout << Sample[sample].InputValues[row][col] << " ";
			cout << endl;
		}
		cout << endl;
		for (int out = 0; out < OutputSize; out++)
			cout << " " << Sample[sample].OutputValues[out] << " ";

		cout << endl << "--------------" << endl;
	}
}
