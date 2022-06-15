/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
* 
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
* 
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer.
* 
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"
#include "Chip.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "Definition.h"

using namespace std;

vector<vector<double> > getNetStructure(const string &inputfile);

int main(int argc, char * argv[]) {   

	auto start = chrono::high_resolution_clock::now();
	
	gen.seed(0);
	
	vector<vector<double> > netStructure;
	netStructure = getNetStructure(argv[2]);
	
	// define weight/input/memory precision from wrapper
	param->synapseBit = atoi(argv[3]);             		 // precision of synapse weight
	param->numBitInput = atoi(argv[4]);            		 // precision of input neural activation
	param->batchSize = atoi(argv[5]);
	param->cellBit = atoi(argv[6]);
	param->technode = atoi(argv[7]);
	param->wireWidth = atoi(argv[8]);
	param->reLu = atoi(argv[9]);
	param->memcelltype = atoi(argv[10]);
	param->levelOutput = atoi(argv[11]);
	param->resistanceOff = 240e3*atoi(argv[12]);

	param->recalculate_Params(param->wireWidth, param->memcelltype);

    // Todo: Change in parse in train.py, pass to hook.py, pass to main.cpp, change offset down there, try printing in some other file
    // Todo: Remove from Param.cpp
    // Todo: IMPORTANT to check all other occurances of the parameter

	// memcelltype: IMPORTANT -> lots of calculations based on this in Param.cpp
	// reLu - change in network as well
	// technode -> change wireWidth accordingly
	// leveloutput corresponding to ADCprecision
	// on/off resistance: in c++ ausrechnen durch faktor und definiertes Verhältnis

	if (param->cellBit > param->synapseBit) {
		cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
		param->cellBit = param->synapseBit;
	}

	/*** initialize operationMode as default ***/
	param->conventionalParallel = 0;
	param->conventionalSequential = 0;
	param->BNNparallelMode = 0;                // parallel BNN
	param->BNNsequentialMode = 0;              // sequential BNN
	param->XNORsequentialMode = 0;           // Use several multi-bit RRAM as one synapse
	param->XNORparallelMode = 0;         // Use several multi-bit RRAM as one synapse
	switch(param->operationmode) {
		case 6:	    param->XNORparallelMode = 1;               break;
		case 5:	    param->XNORsequentialMode = 1;             break;
		case 4:	    param->BNNparallelMode = 1;                break;
		case 3:	    param->BNNsequentialMode = 1;              break;
		case 2:	    param->conventionalParallel = 1;           break;
		case 1:	    param->conventionalSequential = 1;         break;
		case -1:	break;
		default:	exit(-1);
	}

	if (param->XNORparallelMode || param->XNORsequentialMode) {
		param->numRowPerSynapse = 2;
	} else {
		param->numRowPerSynapse = 1;
	}
	if (param->BNNparallelMode) {
		param->numColPerSynapse = 2;
	} else if (param->XNORparallelMode || param->XNORsequentialMode || param->BNNsequentialMode) {
		param->numColPerSynapse = 1;
	} else {
		param->numColPerSynapse = ceil((double)param->synapseBit/(double)param->cellBit);
	}

	switch(param->transistortype) {
		case 3:	    inputParameter.transistorType = TFET;          break;
		case 2:	    inputParameter.transistorType = FET_2D;        break;
		case 1:	    inputParameter.transistorType = conventional;  break;
		case -1:	break;
		default:	exit(-1);
	}

	switch(param->deviceroadmap) {
		case 2:	    inputParameter.deviceRoadmap = LSTP;  break;
		case 1:	    inputParameter.deviceRoadmap = HP;    break;
		case -1:	break;
		default:	exit(-1);
	}

	/* Create SubArray object and link the required global objects (not initialization) */
	inputParameter.temperature = param->temp;   // Temperature (K)
	inputParameter.processNode = param->technode;    // Technology node
	tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);

	double maxPESizeNM, maxTileSizeCM, numPENM;
	vector<int> markNM;
	vector<int> pipelineSpeedUp;
	markNM = ChipDesignInitialize(inputParameter, tech, cell, false, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);
	pipelineSpeedUp = ChipDesignInitialize(inputParameter, tech, cell, true, netStructure, &maxPESizeNM, &maxTileSizeCM, &numPENM);

	double desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM;
	int numTileRow, numTileCol;
	int numArrayWriteParallel;

	vector<vector<double> > numTileEachLayer;
	vector<vector<double> > utilizationEachLayer;
	vector<vector<double> > speedUpEachLayer;
	vector<vector<double> > tileLocaEachLayer;

	numTileEachLayer = ChipFloorPlan(true, false, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);

	utilizationEachLayer = ChipFloorPlan(false, true, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);

	speedUpEachLayer = ChipFloorPlan(false, false, true, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);

	tileLocaEachLayer = ChipFloorPlan(false, false, false, netStructure, markNM,
					maxPESizeNM, maxTileSizeCM, numPENM, pipelineSpeedUp,
					&desiredNumTileNM, &desiredPESizeNM, &desiredNumTileCM, &desiredTileSizeCM, &desiredPESizeCM, &numTileRow, &numTileCol);

	cout << "------------------------------ FloorPlan --------------------------------" <<  endl;
	cout << endl;
	cout << "Tile and PE size are optimized to maximize memory utilization ( = memory mapped by synapse / total memory on chip)" << endl;
	cout << endl;
	if (!param->novelMapping) {
		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
	} else {
		cout << "Desired Conventional Mapped Tile Storage Size: " << desiredTileSizeCM << "x" << desiredTileSizeCM << endl;
		cout << "Desired Conventional PE Storage Size: " << desiredPESizeCM << "x" << desiredPESizeCM << endl;
		cout << "Desired Novel Mapped Tile Storage Size: " << numPENM << "x" << desiredPESizeNM << "x" << desiredPESizeNM << endl;
	}
	cout << "User-defined SubArray Size: " << param->numRowSubArray << "x" << param->numColSubArray << endl;
	cout << endl;
	cout << "----------------- # of tile used for each layer -----------------" <<  endl;
	double totalNumTile = 0;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
		totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
	}
	cout << endl;

	cout << "----------------- Speed-up of each layer ------------------" <<  endl;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << endl;
	}
	cout << endl;

	cout << "----------------- Utilization of each layer ------------------" <<  endl;
	double realMappedMemory = 0;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << utilizationEachLayer[i][0] << endl;
		realMappedMemory += numTileEachLayer[0][i] * numTileEachLayer[1][i] * utilizationEachLayer[i][0];
	}
	cout << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
	cout << endl;
	cout << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
	cout << endl;
	cout << endl;
	cout << endl;

	double numComputation = 0;
	for (int i=0; i<netStructure.size(); i++) {
		numComputation += 2*(netStructure[i][0] * netStructure[i][1] * netStructure[i][2] * netStructure[i][3] * netStructure[i][4] * netStructure[i][5]);
	}

	if (param->trainingEstimation) {
		numComputation *= 3;  // forward, computation of activation gradient, weight gradient
		numComputation -= 2*(netStructure[0][0] * netStructure[0][1] * netStructure[0][2] * netStructure[0][3] * netStructure[0][4] * netStructure[0][5]);  //L-1 does not need AG
		numComputation *= param->batchSize * param->numIteration;  // count for one epoch
	}

	ChipInitialize(inputParameter, tech, cell, netStructure, markNM, numTileEachLayer,
					numPENM, desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, numTileCol, &numArrayWriteParallel);

	double chipHeight, chipWidth, chipArea, chipAreaIC, chipAreaADC, chipAreaAccum, chipAreaOther, chipAreaWG, chipAreaArray;
	double CMTileheight = 0;
	double CMTilewidth = 0;
	double NMTileheight = 0;
	double NMTilewidth = 0;
	vector<double> chipAreaResults;

	chipAreaResults = ChipCalculateArea(inputParameter, tech, cell, desiredNumTileNM, numPENM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow,
					&chipHeight, &chipWidth, &CMTileheight, &CMTilewidth, &NMTileheight, &NMTilewidth);
	chipArea = chipAreaResults[0];
	chipAreaIC = chipAreaResults[1];
	chipAreaADC = chipAreaResults[2];
	chipAreaAccum = chipAreaResults[3];
	chipAreaOther = chipAreaResults[4];
	chipAreaWG = chipAreaResults[5];
	chipAreaArray = chipAreaResults[6];

	double chipReadLatency = 0;
	double chipReadDynamicEnergy = 0;
	double chipReadLatencyAG = 0;
	double chipReadDynamicEnergyAG = 0;
	double chipReadLatencyWG = 0;
	double chipReadDynamicEnergyWG = 0;
	double chipWriteLatencyWU = 0;
	double chipWriteDynamicEnergyWU = 0;

	double chipReadLatencyPeakFW = 0;
	double chipReadDynamicEnergyPeakFW = 0;
	double chipReadLatencyPeakAG = 0;
	double chipReadDynamicEnergyPeakAG = 0;
	double chipReadLatencyPeakWG = 0;
	double chipReadDynamicEnergyPeakWG = 0;
	double chipWriteLatencyPeakWU = 0;
	double chipWriteDynamicEnergyPeakWU = 0;

	double chipLeakageEnergy = 0;
	double chipLeakage = 0;
	double chipbufferLatency = 0;
	double chipbufferReadDynamicEnergy = 0;
	double chipicLatency = 0;
	double chipicReadDynamicEnergy = 0;

	double chipLatencyADC = 0;
	double chipLatencyAccum = 0;
	double chipLatencyOther = 0;
	double chipEnergyADC = 0;
	double chipEnergyAccum = 0;
	double chipEnergyOther = 0;

	double chipDRAMLatency = 0;
	double chipDRAMDynamicEnergy = 0;

	double layerReadLatency = 0;
	double layerReadDynamicEnergy = 0;
	double layerReadLatencyAG = 0;
	double layerReadDynamicEnergyAG = 0;
	double layerReadLatencyWG = 0;
	double layerReadDynamicEnergyWG = 0;
	double layerWriteLatencyWU = 0;
	double layerWriteDynamicEnergyWU = 0;

	double layerReadLatencyPeakFW = 0;
	double layerReadDynamicEnergyPeakFW = 0;
	double layerReadLatencyPeakAG = 0;
	double layerReadDynamicEnergyPeakAG = 0;
	double layerReadLatencyPeakWG = 0;
	double layerReadDynamicEnergyPeakWG = 0;
	double layerWriteLatencyPeakWU = 0;
	double layerWriteDynamicEnergyPeakWU = 0;

	double layerDRAMLatency = 0;
	double layerDRAMDynamicEnergy = 0;

	double tileLeakage = 0;
	double layerbufferLatency = 0;
	double layerbufferDynamicEnergy = 0;
	double layericLatency = 0;
	double layericDynamicEnergy = 0;

	double coreLatencyADC = 0;
	double coreLatencyAccum = 0;
	double coreLatencyOther = 0;
	double coreEnergyADC = 0;
	double coreEnergyAccum = 0;
	double coreEnergyOther = 0;


	if (! param->pipeline) {
		// layer-by-layer process
		// show the detailed hardware performance for each layer
		ofstream layerfile;
            layerfile.open("Layer.csv", ios::out);
            layerfile << "# of Tiles, Speedup, Utilization, readLatency of Forward, readDynamicEnergy of Forward, readLatency of Activation Gradient, " <<
            "readDynamicEnergy of Activation Gradient, readLatency of Weight Gradient, readDynamicEnergy of Weight Gradient, " <<
            "writeLatency of Weight Update, writeDynamicEnergy of Weight Update, PEAK readLatency of Forward, PEAK readDynamicEnergy of Forward, " <<
            "PEAK readLatency of Activation Gradient, PEAK readDynamicEnergy of Activation Gradient, PEAK readLatency of Weight Gradient, " <<
            "PEAK readDynamicEnergy of Weight Gradient, PEAK writeLatency of Weight Update, PEAK writeDynamicEnergy of Weight Update, " <<
            "leakagePower, leakageEnergy, ADC readLatency, Accumulation Circuits readLatency, Synaptic Array w/o ADC readLatency, " <<
            "Buffer buffer latency, Interconnect latency, Weight Gradient Calculation readLatency, Weight Update writeLatency, " <<
            "DRAM data transfer Latency, ADC readDynamicEnergy, Accumulation Circuits readDynamicEnergy, Synaptic Array w/o ADC readDynamicEnergy, " <<
            "Buffer readDynamicEnergy, Interconnect readDynamicEnergy, Weight Gradient Calculation readDynamicEnergy, Weight Update writeDynamicEnergy, " <<
            "DRAM data transfer Energy" << endl;
		for (int i=0; i<netStructure.size(); i++) {
			param->activityRowReadWG = atof(argv[4*i+16]);
            param->activityRowWriteWG = atof(argv[4*i+16]);
            param->activityColWriteWG = atof(argv[4*i+16]);

			ChipCalculatePerformance(inputParameter, tech, cell, i, argv[4*i+13], argv[4*i+14], argv[4*i+15], netStructure[i][6],
						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth, numArrayWriteParallel,
						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerReadLatencyAG, &layerReadDynamicEnergyAG, &layerReadLatencyWG, &layerReadDynamicEnergyWG,
						&layerWriteLatencyWU, &layerWriteDynamicEnergyWU, &layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, &layerDRAMLatency, &layerDRAMDynamicEnergy,
						&layerReadLatencyPeakFW, &layerReadDynamicEnergyPeakFW, &layerReadLatencyPeakAG, &layerReadDynamicEnergyPeakAG,
						&layerReadLatencyPeakWG, &layerReadDynamicEnergyPeakWG, &layerWriteLatencyPeakWU, &layerWriteDynamicEnergyPeakWU);

			double numTileOtherLayer = 0;
			double layerLeakageEnergy = 0;
			for (int j=0; j<netStructure.size(); j++) {
				if (j != i) {
					numTileOtherLayer += numTileEachLayer[0][j] * numTileEachLayer[1][j];
				}
			}
			layerLeakageEnergy = numTileOtherLayer*tileLeakage*(layerReadLatency+layerReadLatencyAG);

            layerfile << numTileEachLayer[0][i] * numTileEachLayer[1][i] << ", ";
            layerfile << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << ", ";
            layerfile << utilizationEachLayer[i][0] << ", ";
			layerfile << layerReadLatency*1e9 << ",";
			layerfile << layerReadDynamicEnergy*1e12 << ",";
			layerfile << layerReadLatencyAG*1e9 << ",";
			layerfile << layerReadDynamicEnergyAG*1e12 << ",";
			layerfile << layerReadLatencyWG*1e9 << ",";
			layerfile << layerReadDynamicEnergyWG*1e12 << ",";
			layerfile << layerWriteLatencyWU*1e9 << ",";
			layerfile << layerWriteDynamicEnergyWU*1e12 << ",";
			layerfile << layerReadLatencyPeakFW*1e9 << ",";
			layerfile << layerReadDynamicEnergyPeakFW*1e12 << ",";
			layerfile << layerReadLatencyPeakAG*1e9 << ",";
			layerfile << layerReadDynamicEnergyPeakAG*1e12 << ",";
			layerfile << layerReadLatencyPeakWG*1e9 << ",";
			layerfile << layerReadDynamicEnergyPeakWG*1e12 << ",";
			layerfile << layerWriteLatencyPeakWU*1e9 << ",";
			layerfile << layerWriteDynamicEnergyPeakWU*1e12 << ",";
			layerfile << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << ",";
			layerfile << layerLeakageEnergy*1e12 << ",";

			layerfile << coreLatencyADC*1e9 << ",";
			layerfile << coreLatencyAccum*1e9 << ",";
			layerfile << coreLatencyOther*1e9 << ",";
			layerfile << layerbufferLatency*1e9 << ",";
			layerfile << layericLatency*1e9 << ",";
			layerfile << layerReadLatencyPeakWG*1e9 << ",";
			layerfile << layerWriteLatencyPeakWU*1e9 << ",";
			layerfile << layerDRAMLatency*1e9 << ",";
			layerfile << coreEnergyADC*1e12 << ",";
			layerfile << coreEnergyAccum*1e12 << ",";
			layerfile << coreEnergyOther*1e12 << ",";
			layerfile << layerbufferDynamicEnergy*1e12 << ",";
			layerfile << layericDynamicEnergy*1e12 << ",";
			layerfile << layerReadDynamicEnergyPeakWG*1e12 << ",";
			layerfile << layerWriteDynamicEnergyPeakWU*1e12 << ",";
			layerfile << layerDRAMDynamicEnergy*1e12 << endl;



			chipReadLatency += layerReadLatency;
			chipReadDynamicEnergy += layerReadDynamicEnergy;
			chipReadLatencyAG += layerReadLatencyAG;
			chipReadDynamicEnergyAG += layerReadDynamicEnergyAG;
			chipReadLatencyWG += layerReadLatencyWG;
			chipReadDynamicEnergyWG += layerReadDynamicEnergyWG;
			chipWriteLatencyWU += layerWriteLatencyWU;
			chipWriteDynamicEnergyWU += layerWriteDynamicEnergyWU;
			chipDRAMLatency += layerDRAMLatency;
			chipDRAMDynamicEnergy += layerDRAMDynamicEnergy;

			chipReadLatencyPeakFW += layerReadLatencyPeakFW;
			chipReadDynamicEnergyPeakFW += layerReadDynamicEnergyPeakFW;
			chipReadLatencyPeakAG += layerReadLatencyPeakAG;
			chipReadDynamicEnergyPeakAG += layerReadDynamicEnergyPeakAG;
			chipReadLatencyPeakWG += layerReadLatencyPeakWG;
			chipReadDynamicEnergyPeakWG += layerReadDynamicEnergyPeakWG;
			chipWriteLatencyPeakWU += layerWriteLatencyPeakWU;
			chipWriteDynamicEnergyPeakWU += layerWriteDynamicEnergyPeakWU;

			chipLeakageEnergy += layerLeakageEnergy;
			chipLeakage += tileLeakage*numTileEachLayer[0][i] * numTileEachLayer[1][i];
			chipbufferLatency += layerbufferLatency;
			chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
			chipicLatency += layericLatency;
			chipicReadDynamicEnergy += layericDynamicEnergy;

			chipLatencyADC += coreLatencyADC;
			chipLatencyAccum += coreLatencyAccum;
			chipLatencyOther += coreLatencyOther;
			chipEnergyADC += coreEnergyADC;
			chipEnergyAccum += coreEnergyAccum;
			chipEnergyOther += coreEnergyOther;
		}
		layerfile.close();
	} else {
		// pipeline system
		// firstly define system clock
		double systemClock = 0;
		double systemClockAG = 0;
		double systemClockPeakFW = 0;
		double systemClockPeakAG = 0;

		vector<double> readLatencyPerLayer;
		vector<double> readDynamicEnergyPerLayer;
		vector<double> readLatencyPerLayerAG;
		vector<double> readDynamicEnergyPerLayerAG;
		vector<double> readLatencyPerLayerWG;
		vector<double> readDynamicEnergyPerLayerWG;
		vector<double> writeLatencyPerLayerWU;
		vector<double> writeDynamicEnergyPerLayerWU;

		vector<double> readLatencyPerLayerPeakFW;
		vector<double> readDynamicEnergyPerLayerPeakFW;
		vector<double> readLatencyPerLayerPeakAG;
		vector<double> readDynamicEnergyPerLayerPeakAG;
		vector<double> readLatencyPerLayerPeakWG;
		vector<double> readDynamicEnergyPerLayerPeakWG;
		vector<double> writeLatencyPerLayerPeakWU;
		vector<double> writeDynamicEnergyPerLayerPeakWU;

		vector<double> dramLatencyPerLayer;
		vector<double> dramDynamicEnergyPerLayer;

		vector<double> leakagePowerPerLayer;
		vector<double> bufferLatencyPerLayer;
		vector<double> bufferEnergyPerLayer;
		vector<double> icLatencyPerLayer;
		vector<double> icEnergyPerLayer;

		vector<double> coreLatencyADCPerLayer;
		vector<double> coreEnergyADCPerLayer;
		vector<double> coreLatencyAccumPerLayer;
		vector<double> coreEnergyAccumPerLayer;
		vector<double> coreLatencyOtherPerLayer;
		vector<double> coreEnergyOtherPerLayer;

		for (int i=0; i<netStructure.size(); i++) {

            param->activityRowReadWG = atof(argv[4*i+16]);
            param->activityRowWriteWG = atof(argv[4*i+16]);
            param->activityColWriteWG = atof(argv[4*i+16]);
			ChipCalculatePerformance(inputParameter, tech, cell, i, argv[4*i+13], argv[4*i+14], argv[4*i+15], netStructure[i][6],
						netStructure, markNM, numTileEachLayer, utilizationEachLayer, speedUpEachLayer, tileLocaEachLayer,
						numPENM, desiredPESizeNM, desiredTileSizeCM, desiredPESizeCM, CMTileheight, CMTilewidth, NMTileheight, NMTilewidth, numArrayWriteParallel,
						&layerReadLatency, &layerReadDynamicEnergy, &tileLeakage, &layerReadLatencyAG, &layerReadDynamicEnergyAG, &layerReadLatencyWG, &layerReadDynamicEnergyWG, &layerWriteLatencyWU, &layerWriteDynamicEnergyWU,
						&layerbufferLatency, &layerbufferDynamicEnergy, &layericLatency, &layericDynamicEnergy,
						&coreLatencyADC, &coreLatencyAccum, &coreLatencyOther, &coreEnergyADC, &coreEnergyAccum, &coreEnergyOther, &layerDRAMLatency, &layerDRAMDynamicEnergy,
						&layerReadLatencyPeakFW, &layerReadDynamicEnergyPeakFW, &layerReadLatencyPeakAG, &layerReadDynamicEnergyPeakAG,
						&layerReadLatencyPeakWG, &layerReadDynamicEnergyPeakWG, &layerWriteLatencyPeakWU, &layerWriteDynamicEnergyPeakWU);


			systemClock = MAX(systemClock, layerReadLatency);
			systemClockAG = MAX(systemClockAG, layerReadLatencyAG);
			systemClockPeakFW = MAX(systemClockPeakFW, layerReadLatencyPeakFW);
			systemClockPeakAG = MAX(systemClockPeakAG, layerReadLatencyPeakAG);
			chipLatencyADC = MAX(chipLatencyADC, coreLatencyADCPerLayer[i]);
			chipLatencyAccum = MAX(chipLatencyAccum, coreLatencyAccumPerLayer[i]);
			chipLatencyOther = MAX(chipLatencyOther, coreLatencyOtherPerLayer[i]);

			readLatencyPerLayer.push_back(layerReadLatency);
			readDynamicEnergyPerLayer.push_back(layerReadDynamicEnergy);
			readLatencyPerLayerAG.push_back(layerReadLatencyAG);
			readDynamicEnergyPerLayerAG.push_back(layerReadDynamicEnergyAG);
			readLatencyPerLayerWG.push_back(layerReadLatencyWG);
			readDynamicEnergyPerLayerWG.push_back(layerReadDynamicEnergyWG);
			writeLatencyPerLayerWU.push_back(layerWriteLatencyWU);
			writeDynamicEnergyPerLayerWU.push_back(layerWriteDynamicEnergyWU);
			dramLatencyPerLayer.push_back(layerDRAMLatency);
			dramDynamicEnergyPerLayer.push_back(layerDRAMDynamicEnergy);

			readLatencyPerLayerPeakFW.push_back(layerReadLatencyPeakFW);
			readDynamicEnergyPerLayerPeakFW.push_back(layerReadDynamicEnergyPeakFW);
			readLatencyPerLayerPeakAG.push_back(layerReadLatencyPeakAG);
			readDynamicEnergyPerLayerPeakAG.push_back(layerReadDynamicEnergyPeakAG);
			readLatencyPerLayerPeakWG.push_back(layerReadLatencyPeakWG);
			readDynamicEnergyPerLayerPeakWG.push_back(layerReadDynamicEnergyPeakWG);
			writeLatencyPerLayerPeakWU.push_back(layerWriteLatencyPeakWU);
			writeDynamicEnergyPerLayerPeakWU.push_back(layerWriteDynamicEnergyPeakWU);

			leakagePowerPerLayer.push_back(numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage);
			bufferLatencyPerLayer.push_back(layerbufferLatency);
			bufferEnergyPerLayer.push_back(layerbufferDynamicEnergy);
			icLatencyPerLayer.push_back(layericLatency);
			icEnergyPerLayer.push_back(layericDynamicEnergy);

			coreLatencyADCPerLayer.push_back(coreLatencyADC);
			coreEnergyADCPerLayer.push_back(coreEnergyADC);
			coreLatencyAccumPerLayer.push_back(coreLatencyAccum);
			coreEnergyAccumPerLayer.push_back(coreEnergyAccum);
			coreLatencyOtherPerLayer.push_back(coreLatencyOther);
			coreEnergyOtherPerLayer.push_back(coreEnergyOther);

			chipReadDynamicEnergy += layerReadDynamicEnergy;
			chipReadDynamicEnergyAG += layerReadDynamicEnergyAG;
			chipReadDynamicEnergyWG += layerReadDynamicEnergyWG;
			chipWriteDynamicEnergyWU += layerWriteDynamicEnergyWU;
			// since Weight Gradient and Weight Update have limitation on hardware resource, do not implement pipeline
			chipReadLatencyWG += layerReadLatencyWG;
			chipWriteLatencyWU += layerWriteLatencyWU;

			chipReadDynamicEnergyPeakFW += layerReadDynamicEnergyPeakFW;
			chipReadDynamicEnergyPeakAG += layerReadDynamicEnergyPeakAG;
			chipReadDynamicEnergyPeakWG += layerReadDynamicEnergyPeakWG;
			chipWriteDynamicEnergyPeakWU += layerWriteDynamicEnergyPeakWU;

			chipDRAMLatency += layerDRAMLatency;
			chipDRAMDynamicEnergy += layerDRAMDynamicEnergy;

			chipLeakage += numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage;
			chipbufferLatency = MAX(chipbufferLatency, layerbufferLatency);
			chipbufferReadDynamicEnergy += layerbufferDynamicEnergy;
			chipicLatency = MAX(chipicLatency, layericLatency);
			chipicReadDynamicEnergy += layericDynamicEnergy;
			chipEnergyADC += coreEnergyADC;
			chipEnergyAccum += coreEnergyAccum;
			chipEnergyOther += coreEnergyOther;

		}
		chipReadLatency = systemClock;
		chipReadLatencyAG = systemClockAG;
		chipReadLatencyPeakFW = systemClockPeakFW;
		chipReadLatencyPeakAG = systemClockPeakAG;

        ofstream layerfile;
        layerfile.open("Layer.csv", ios::out);
        layerfile << "# of Tiles, Speedup, Utilization, readLatency, readDynamicEnergy, readLatency of Activation Gradient, readDynamicEnergy of Activation Gradient, " <<
        "readLatency of Weight Gradient, readDynamicEnergy of Weight Gradient, writeLatency of Weight Update, writeDynamicEnergy of Weight Update, " <<
        "PEAK readLatency, PEAK readDynamicEnergy, PEAK readLatency of Activation Gradient, PEAK readDynamicEnergy of Activation Gradient, " <<
        "PEAK readLatency of Weight Gradient, PEAK readDynamicEnergy of Weight Gradient, PEAK writeLatency of Weight Update, " <<
        "PEAK writeDynamicEnergy of Weight Update, leakagePower, leakageEnergy, ADC readLatency, Accumulation Circuits readLatency, " <<
        "Synaptic Array w/o ADC readLatency, Buffer latency, Interconnect latency, Weight Gradient Calculation readLatencyWeight Gradient Calculation readLatency, " <<
        "Weight Update writeLatency, DRAM data transfer Latency, ADC readDynamicEnergy, Accumulation Circuits readDynamicEnergy, " <<
        "Synaptic Array w/o ADC readDynamicEnergy, Buffer readDynamicEnergy, Interconnect readDynamicEnergy, Weight Gradient Calculation readDynamicEnergy, " <<
        "Weight Update writeDynamicEnergy, DRAM data transfer DynamicEnergy" << endl;
		for (int i=0; i<netStructure.size(); i++) {
            // Build layer estimation csv file, one row for each layer
			cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;

            layerfile << numTileEachLayer[0][i] * numTileEachLayer[1][i] << ", ";
            layerfile << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << ", ";
            layerfile << utilizationEachLayer[i][0] << ", ";
			layerfile << readLatencyPerLayer[i]*1e9 << ",";
			layerfile << readDynamicEnergyPerLayer[i]*1e12 << ",";
			layerfile << readLatencyPerLayerAG[i]*1e9 << ",";
			layerfile << readDynamicEnergyPerLayerAG[i]*1e12 << ",";
			layerfile << readLatencyPerLayerWG[i]*1e9 << ",";
			layerfile << readDynamicEnergyPerLayerWG[i]*1e12 << ",";
			layerfile << writeLatencyPerLayerWU[i]*1e9 << ",";
			layerfile << writeDynamicEnergyPerLayerWU[i]*1e12 << ",";
			layerfile << readLatencyPerLayerPeakFW[i]*1e9 << ",";
			layerfile << readDynamicEnergyPerLayerPeakFW[i]*1e12 << ",";
			layerfile << readLatencyPerLayerPeakAG[i]*1e9 << ",";
			layerfile << readDynamicEnergyPerLayerPeakAG[i]*1e12 << ",";
			layerfile << readLatencyPerLayerPeakWG[i]*1e9 << ",";
			layerfile << readDynamicEnergyPerLayerPeakWG[i]*1e12 << ",";
			layerfile << writeLatencyPerLayerPeakWU[i]*1e9 << ",";
			layerfile << writeDynamicEnergyPerLayerPeakWU[i]*1e12 << ",";
			layerfile << leakagePowerPerLayer[i]*1e6 << ",";
			layerfile << leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]) *1e12 << ",";
			layerfile << coreLatencyADCPerLayer[i]*1e9 << ",";
			layerfile << coreLatencyAccumPerLayer[i]*1e9 << ",";
			layerfile << coreLatencyOtherPerLayer[i]*1e9 << ",";
			layerfile << bufferLatencyPerLayer[i]*1e9 << ",";
			layerfile << icLatencyPerLayer[i]*1e9 << ",";
			layerfile << readLatencyPerLayerPeakWG[i]*1e9 << ",";
			layerfile << writeLatencyPerLayerPeakWU[i]*1e9 << ",";
			layerfile << dramLatencyPerLayer[i]*1e9 << ",";
			layerfile << coreEnergyADCPerLayer[i]*1e12 << ",";
			layerfile << coreEnergyAccumPerLayer[i]*1e12 << ",";
			layerfile << coreEnergyOtherPerLayer[i]*1e12 << ",";
			layerfile << bufferEnergyPerLayer[i]*1e12 << ",";
			layerfile << icEnergyPerLayer[i]*1e12 << ",";
			layerfile << readDynamicEnergyPerLayerPeakWG[i]*1e12 << ",";
			layerfile << writeDynamicEnergyPerLayerPeakWU[i]*1e12 << ",";
			layerfile << dramDynamicEnergyPerLayer[i]*1e12 << endl;

			chipLeakageEnergy += leakagePowerPerLayer[i] * ((systemClock-readLatencyPerLayer[i]) + (systemClockAG-readLatencyPerLayerAG[i]));


		}
		layerfile.close();
	}



    ofstream summaryfile;
    summaryfile.open("Summary.csv", ios::out);
    summaryfile << "Utilization, Chip Area, Chip total CIM array, Total IC Area on chip, Total ADC Area on chip, Total Accumulation Circuits on chip, " <<
      "Other Peripheries, Weight Gradient Calculation, Chip readLatency of Forward, Chip readDynamicEnergy of Forward, " <<
      "Chip readLatency of Activation Gradient, Chip readDynamicEnergy of Activation Gradient, Chip readLatency of Weight Gradient, " <<
      "Chip readDynamicEnergy of Weight Gradient, Chip writeLatency of Weight Update, Chip writeDynamicEnergy of Weight Update, " <<
      "Chip total Latency, Chip total Energy, Chip PEAK readLatency of Forward, Chip PEAK readDynamicEnergy of Forward, " <<
      "Chip PEAK readLatency of Activation Gradient, Chip PEAK readDynamicEnergy of Activation Gradient, Chip PEAK readLatency of Weight Gradient, " <<
      "Chip PEAK readDynamicEnergy of Weight Gradient, Chip PEAK writeLatency of Weight Update, Chip PEAK writeDynamicEnergy of Weight Update, " <<
      "Chip PEAK total Latency, Chip PEAK total Energy, Chip leakage Energy, Chip leakage Power, ADC readLatency, " <<
      "Accumulation Circuits readLatency, Synaptic Array w/o ADC readLatency, Buffer readLatency, Interconnect readLatency, " <<
      "Weight Gradient Calculation readLatency, Weight Update writeLatency, DRAM data transfer Latency, ADC readDynamicEnergy, " <<
      "Accumulation Circuits readDynamicEnergy, Synaptic Array w/o ADC readDynamicEnergy, Buffer readDynamicEnergy, " <<
      "Interconnect readDynamicEnergy, Weight Gradient Calculation readDynamicEnergy, Weight Update writeDynamicEnergy, " <<
      "DRAM data transfer DynamicEnergy, Energy Efficiency TOPS/W, Throughput TOPS, Throughput FPS, Peak Energy Efficiency TOPS/W, " <<
      "Peak Throughput TOPS, Peak Throughput FPS" << endl;

    summaryfile << realMappedMemory/totalNumTile*100 << ", ";
	summaryfile << chipArea*1e12 << ",";
	summaryfile << chipAreaArray*1e12 << ",";
	summaryfile << chipAreaIC*1e12 << ",";
	summaryfile << chipAreaADC*1e12 << ",";
	summaryfile << chipAreaAccum*1e12 << ",";
	summaryfile << chipAreaOther*1e12 << ",";
	summaryfile << chipAreaWG*1e12 << ",";
	cout << endl;
	if (! param->pipeline) {
	} else {
	}
	summaryfile << chipReadLatency*1e9 << ",";
	summaryfile << chipReadDynamicEnergy*1e12 << ",";
	summaryfile << chipReadLatencyAG*1e9 << ",";
	summaryfile << chipReadDynamicEnergyAG*1e12 << ",";
	summaryfile << chipReadLatencyWG*1e9 << ",";
	summaryfile << chipReadDynamicEnergyWG*1e12 << ",";
	summaryfile << chipWriteLatencyWU*1e9 << ",";
	summaryfile << chipWriteDynamicEnergyWU*1e12 << ",";
	summaryfile << (chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e9 << ",";
	summaryfile << (chipReadDynamicEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU) << ",";
	summaryfile << chipReadLatencyPeakFW*1e9 << ",";
	summaryfile << chipReadDynamicEnergyPeakFW*1e12 << ",";
	summaryfile << chipReadLatencyPeakAG*1e9 << ",";
	summaryfile << chipReadDynamicEnergyPeakAG*1e12 << ",";
	summaryfile << chipReadLatencyPeakWG*1e9 << ",";
	summaryfile << chipReadDynamicEnergyPeakWG*1e12 << ",";
	summaryfile << chipWriteLatencyPeakWU*1e9 << ",";
	summaryfile << chipWriteDynamicEnergyPeakWU*1e12 << ",";
	summaryfile << (chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e9 << ",";
	summaryfile << (chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*1e12 << ",";
	summaryfile << chipLeakageEnergy*1e12 << ",";
	summaryfile << chipLeakage*1e6 << ",";
	summaryfile << chipLatencyADC*1e9 << ",";
	summaryfile << chipLatencyAccum*1e9 << ",";
	summaryfile << chipLatencyOther*1e9 << ",";
	summaryfile << chipbufferLatency*1e9 << ",";
	summaryfile << chipicLatency*1e9 << ",";
	summaryfile << chipReadLatencyPeakWG*1e9 << ",";
	summaryfile << chipWriteLatencyPeakWU*1e9 << ",";
	summaryfile << chipDRAMLatency*1e9 << ",";
	summaryfile << chipEnergyADC*1e12 << ",";
	summaryfile << chipEnergyAccum*1e12 << ",";
	summaryfile << chipEnergyOther*1e12 << ",";
	summaryfile << chipbufferReadDynamicEnergy*1e12 << ",";
	summaryfile << chipicReadDynamicEnergy*1e12 << ",";
	summaryfile << chipReadDynamicEnergyPeakWG*1e12 << ",";
	summaryfile << chipWriteDynamicEnergyPeakWU*1e12 << ",";
	summaryfile << chipDRAMDynamicEnergy*1e12 << ",";

	if (! param->pipeline) {
	} else {
	}

	summaryfile << numComputation/((chipReadDynamicEnergy+chipLeakageEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*1e12) << ",";
	summaryfile << numComputation/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e-12 << ",";
	summaryfile << 1/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU) << ",";
	summaryfile << numComputation/((chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*1e12) << ",";
	summaryfile << numComputation/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e-12 << ",";
    summaryfile << 1/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU) << endl;

	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::seconds>(stop-start);
    cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
	cout << "Total Run-time of NeuroSim: " << duration.count() << " seconds" << endl;
	cout << "------------------------------ Simulation Performance --------------------------------" <<  endl;
    summaryfile.close();


	
	
	return 0;
}

vector<vector<double> > getNetStructure(const string &inputfile) {
	ifstream infile(inputfile.c_str());      
	string inputline;
	string inputval;
	
	int ROWin=0, COLin=0;      
	if (!infile.good()) {        
		cerr << "Error: the input file cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(infile, inputline, '\n')) {       
			ROWin++;                                
		}
		infile.clear();
		infile.seekg(0, ios::beg);      
		if (getline(infile, inputline, '\n')) {        
			istringstream iss (inputline);      
			while (getline(iss, inputval, ',')) {       
				COLin++;
			}
		}	
	}
	infile.clear();
	infile.seekg(0, ios::beg);          

	vector<vector<double> > netStructure;               
	for (int row=0; row<ROWin; row++) {	
		vector<double> netStructurerow;
		getline(infile, inputline, '\n');             
		istringstream iss;
		iss.str(inputline);
		for (int col=0; col<COLin; col++) {       
			while(getline(iss, inputval, ',')){	
				istringstream fs;
				fs.str(inputval);
				double f=0;
				fs >> f;				
				netStructurerow.push_back(f);			
			}			
		}		
		netStructure.push_back(netStructurerow);
	}
	infile.close();
	
	return netStructure;
	netStructure.clear();
}	



