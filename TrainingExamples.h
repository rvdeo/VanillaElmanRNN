#ifndef TRAININGEXAMPLES_H
#define TRAININGEXAMPLES_H

#include "Samples.h"

typedef vector<Samples> DataSample;

class TrainingExamples {
public:

	char* FileName;

	int SampleSize;
	int ColumnSize;
	int OutputSize;
	int RowSize;

	DataSample Sample;

	int row;
    int col;
    int layer;
    int r;
    int x;
    int y;


public:
	TrainingExamples();

	TrainingExamples(char* File, int sampleSize, int columnSize,
			int outputSize) ;
	void printData();

};

#endif // TRAININGEXAMPLES_H
