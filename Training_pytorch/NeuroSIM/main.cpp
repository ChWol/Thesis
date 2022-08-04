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

	param->synapseBit = atoi(argv[3]);
	param->numBitInput = atoi(argv[4]);
	param->batchSize = atoi(argv[5]);
	param->cellBit = atoi(argv[6]);
	param->technode = atoi(argv[7]);
	param->wireWidth = atoi(argv[8]);
	param->reLu = atoi(argv[9]);
	param->memcelltype = atoi(argv[10]);
	param->levelOutput = atoi(argv[11]);
	param->resistanceOff = 240e3*atoi(argv[12]);
	param->rule = argv[13];

	param->recalculate_Params(param->wireWidth, param->memcelltype, param->resistanceOff);

	cout << param->synapseBit << endl;
	cout << param->numBitInput << endl;
	cout << param->batchSize << endl;
	cout << param->cellBit << endl;
	cout << param->technode << endl;
	cout << param->wireWidth << endl;
	cout << param->reLu << endl;
	cout << param->memcelltype << endl;
	cout << param->levelOutput << endl;
	cout << param->resistanceOff << endl;
	cout << param->unitLengthWireResistance << endl;

	if (param->cellBit > param->synapseBit) {
		cout << "ERROR!: Memory precision is even higher than synapse precision, please modify 'cellBit' in Param.cpp!" << endl;
		param->cellBit = param->synapseBit;
	}

	// My addition
	double max_layer_output = 0;
	double num_classes = netStructure[netStructure.size()-1][5];
	for (int i=0; i<netStructure.size(); i++) {
	    if (netStructure[i][5] > max_layer_output) {
	        max_layer_output = netStructure[i][5];
	    }
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

	// My addition
	double dfaTiles = 0;
	double dfaRealMappedMemory = 0;
	if (param->rule == "dfa") {
	    double dfaTileRows = ceil(max_layer_output*(double) param->numRowPerSynapse/(double) desiredTileSizeCM);
        double dfaTileColumns = ceil(num_classes*(double) param->numColPerSynapse/(double) desiredTileSizeCM);
        dfaTiles = dfaTileRows*dfaTileColumns;

        double utilization = (max_layer_output*param->numRowPerSynapse*num_classes*param->numColPerSynapse)/(dfaTiles*desiredTileSizeCM*desiredTileSizeCM);
        dfaRealMappedMemory = dfaTiles*utilization;
    }

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
	cout << endl;

	double totalNumTile = 0;
	for (int i=0; i<netStructure.size(); i++) {
		cout << "layer" << i+1 << ": " << numTileEachLayer[0][i] * numTileEachLayer[1][i] << endl;
		totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
	}
	// My addition
	totalNumTile += dfaTiles;

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
	// My addition
	realMappedMemory += dfaRealMappedMemory;

	cout << "Memory Utilization of Whole Chip: " << realMappedMemory/totalNumTile*100 << " % " << endl;
	cout << endl;
	cout << "---------------------------- FloorPlan Done ------------------------------" <<  endl;
	cout << endl;
	cout << endl;
	cout << endl;

    // My addition
	double numComputation = 0;
	double numComputation_Forward = 0;
	for (int i=0; i<netStructure.size(); i++) {
		numComputation_Forward += 2*(netStructure[i][0] * netStructure[i][1] * netStructure[i][2] * netStructure[i][3] * netStructure[i][4] * netStructure[i][5]);
	}

    double numComputation_BP = 0;
	if (param->trainingEstimation) {
		numComputation_BP = 2 * numComputation_Forward;
		numComputation_BP -= 2*(netStructure[0][0] * netStructure[0][1] * netStructure[0][2] * netStructure[0][3] * netStructure[0][4] * netStructure[0][5]);
	}

    double  numComputation_DFA = 0;
	if (param->trainingEstimation) {
	    numComputation_DFA = 1 * numComputation_Forward;
	    numComputation_DFA -= 2*(netStructure[0][0] * netStructure[0][1] * netStructure[0][2] * netStructure[0][3] * netStructure[0][4] * netStructure[0][5]);
	    for (int i=0; i<netStructure.size(); i++) {
		    numComputation_DFA += 2*(netStructure[i][0] * netStructure[i][1] * num_classes * netStructure[i][3] * netStructure[i][4] * netStructure[i][5]);
	    }
	}

	double scalingFactor_Total = 1;
	double scalingFactor_WG = 1;
	if (param->rule == "dfa") {
	    scalingFactor_Total = 1 - (numComputation_BP - numComputation_DFA) / (numComputation_Forward + numComputation_BP);
	    scalingFactor_WG = 1 - (numComputation_BP - numComputation_DFA) / (numComputation_BP);
	}

    if (param->rule == "bp") {
        numComputation = numComputation_Forward + numComputation_BP;
    }
    else {
        numComputation = numComputation_Forward + numComputation_DFA;
    }
	numComputation *= param->batchSize * param->numIteration;
	// End of my addition

	// Alternative approach
	double flopsBP = 0;
	for (int i=0; i<netStructure.size(); i++) {
	    if (i == netStructure.size() - 1) {
	        flopsBP += netStructure[i][5]*(netStructure[i][2]-1)*2*param->batchSize + netStructure[i][5] * param->batchSize + netStructure[i][5]*2*(param->batchSize-1)*netStructure[i][2] + netStructure[i][5] * param->batchSize;
	    }
	    else {
	        flopsBP += num_classes * 2 * (param->batchSize-1) * netStructure[i][2] + numComputation_Forward;
	    }
	}

	double flopsDFA = 0;
	for (int i=0; i<netStructure.size(); i++) {
	    if (i == netStructure.size() - 1) {
	        flopsDFA += netStructure[i][5]*2*(num_classes-1)*param->batchSize + netStructure[i][5] * param->batchSize + netStructure[i][5]*2*(param->batchSize-1)*netStructure[i][2] + netStructure[i][5] * param->batchSize;
	    }
	    else {
	        flopsBP += num_classes * 2 * (param->batchSize-1) * netStructure[i][2] + numComputation_Forward;
	    }
	}

    cout << "Approximation via forward pass" << endl;
	cout << "BP: " << numComputation_BP << endl;
	cout << "DFA: " << numComputation_DFA << endl;
	cout << "Scaling factor: " << scalingFactor_WG << endl;
	cout << endl;
    cout << "FLOPs approach" << endl;
	cout << "BP: " << flopsBP << endl;
	cout << "DFA: " << flopsDFA << endl;
	cout << "Scaling factor: " << 1- (flopsBP - flopsDFA) / (flopsBP) << endl;

	cout << endl;
	cout << "Total: " << scalingFactor_Total << endl;

	ChipInitialize(inputParameter, tech, cell, netStructure, markNM, numTileEachLayer,
					numPENM, desiredNumTileNM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow, numTileCol, &numArrayWriteParallel);

	double chipHeight, chipWidth, chipArea, chipAreaIC, chipAreaADC, chipAreaAccum, chipAreaOther, chipAreaWG, chipAreaArray;
	double CMTileheight = 0;
	double CMTilewidth = 0;
	double NMTileheight = 0;
	double NMTilewidth = 0;
	vector<double> chipAreaResults;

	chipAreaResults = ChipCalculateArea(inputParameter, tech, cell, max_layer_output, num_classes, netStructure.size(), desiredNumTileNM, numPENM, desiredPESizeNM, desiredNumTileCM, desiredTileSizeCM, desiredPESizeCM, numTileRow,
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


	cout << "-------------------------------------- Hardware Performance --------------------------------------" <<  endl;

	if (! param->pipeline) {
		ofstream layerfile;
            layerfile.open("Layer.csv", ios::out);
            layerfile << "# of Tiles, Speed-up, Utilization, Read Latency of Forward (ns), Read Dynamic Energy of Forward (pJ), Read Latency of Activation Gradient (ns), " <<
            "Read Dynamic Energy of Activation Gradient (pJ), Read Latency of Weight Gradient (ns), Read Dynamic Energy of Weight Gradient (pJ), " <<
            "Write Latency of Weight Update (ns), Write Dynamic Energy of Weight Update (pJ), PEAK Read Latency of Forward (ns), PEAK Read Dynamic Energy of Forward (pJ), " <<
            "PEAK Read Latency of Activation Gradient (ns), PEAK Read Dynamic Energy of Activation Gradient (pJ), PEAK Read Latency of Weight Gradient (ns), " <<
            "PEAK Read Dynamic Energy of Weight Gradient (pJ), PEAK Write Latency of Weight Update (ns), PEAK Write Dynamic Energy of Weight Update (pJ), " <<
            "Leakage Power (uW), Leakage Energy (pJ), ADC Read Latency (ns), Accumulation Circuits Read Latency (ns), Synaptic Array w/o ADC Read Latency (ns), " <<
            "Buffer Buffer Latency (ns), Interconnect Latency (ns), Weight Gradient Calculation Read Latency (ns), Weight Update Write Latency (ns), " <<
            "DRAM data transfer Latency (ns), ADC Read Dynamic Energy (pJ), Accumulation Circuits Read Dynamic Energy (pJ), Synaptic Array w/o ADC Read Dynamic Energy (pJ), " <<
            "Buffer Read Dynamic Energy (pJ), Interconnect Read Dynamic Energy (pJ), Weight Gradient Calculation Read Dynamic Energy (pJ), Weight Update Write Dynamic Energy (pJ), " <<
            "DRAM data transfer Energy (pJ)" << endl;
		for (int i=0; i<netStructure.size(); i++) {
			cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;
			param->activityRowReadWG = atof(argv[4*i+17]);
            param->activityRowWriteWG = atof(argv[4*i+17]);
            param->activityColWriteWG = atof(argv[4*i+17]);

			ChipCalculatePerformance(inputParameter, tech, cell, i, argv[4*i+14], argv[4*i+15], argv[4*i+16], netStructure[i][6],
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
			cout << "layer" << i+1 << "'s readLatency of Forward is: " << layerReadLatency*1e9 << "ns" << endl;
			layerfile << layerReadLatency*1e9 << ",";
			cout << "layer" << i+1 << "'s readDynamicEnergy of Forward is: " << layerReadDynamicEnergy*1e12 << "pJ" << endl;
			layerfile << layerReadDynamicEnergy*1e12 << ",";
			cout << "layer" << i+1 << "'s readLatency of Activation Gradient is: " << layerReadLatencyAG*1e9 << "ns" << endl;
			layerfile << layerReadLatencyAG*1e9 << ",";
			cout << "layer" << i+1 << "'s readDynamicEnergy of Activation Gradient is: " << layerReadDynamicEnergyAG*1e12 << "pJ" << endl;
			layerfile << layerReadDynamicEnergyAG*1e12 << ",";
			cout << "layer" << i+1 << "'s readLatency of Weight Gradient is: " << layerReadLatencyWG*1e9 << "ns" << endl;
			layerfile << layerReadLatencyWG*1e9 << ",";
			// My addition
			cout << "layer" << i+1 << "'s readDynamicEnergy of Weight Gradient is: " << layerReadDynamicEnergyWG*scalingFactor_WG*1e12<< "pJ" << endl;
			layerfile << layerReadDynamicEnergyWG*scalingFactor_WG*1e12 << ",";
			cout << "layer" << i+1 << "'s writeLatency of Weight Update is: " << layerWriteLatencyWU*1e9 << "ns" << endl;
			layerfile << layerWriteLatencyWU*1e9 << ",";
			cout << "layer" << i+1 << "'s writeDynamicEnergy of Weight Update is: " << layerWriteDynamicEnergyWU*1e12 << "pJ" << endl;
			layerfile << layerWriteDynamicEnergyWU*1e12 << ",";
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "layer" << i+1 << "'s PEAK readLatency of Forward is: " << layerReadLatencyPeakFW*1e9 << "ns" << endl;
			layerfile << layerReadLatencyPeakFW*1e9 << ",";
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Forward is: " << layerReadDynamicEnergyPeakFW*1e12 << "pJ" << endl;
			layerfile << layerReadDynamicEnergyPeakFW*1e12 << ",";
			cout << "layer" << i+1 << "'s PEAK readLatency of Activation Gradient is: " << layerReadLatencyPeakAG*1e9 << "ns" << endl;
			layerfile << layerReadLatencyPeakAG*1e9 << ",";
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Activation Gradient is: " << layerReadDynamicEnergyPeakAG*1e12 << "pJ" << endl;
			layerfile << layerReadDynamicEnergyPeakAG*1e12 << ",";
			cout << "layer" << i+1 << "'s PEAK readLatency of Weight Gradient is: " << layerReadLatencyPeakWG*1e9 << "ns" << endl;
			layerfile << layerReadLatencyPeakWG*1e9 << ",";
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Weight Gradient is: " << layerReadDynamicEnergyPeakWG*scalingFactor_WG*1e12 << "pJ" << endl;
			layerfile << layerReadDynamicEnergyPeakWG*scalingFactor_WG*1e12 << ",";
			cout << "layer" << i+1 << "'s PEAK writeLatency of Weight Update is: " << layerWriteLatencyPeakWU*1e9 << "ns" << endl;
			layerfile << layerWriteLatencyPeakWU*1e9 << ",";
			cout << "layer" << i+1 << "'s PEAK writeDynamicEnergy of Weight Update is: " << layerWriteDynamicEnergyPeakWU*1e12 << "pJ" << endl;
			layerfile << layerWriteDynamicEnergyPeakWU*1e12 << ",";
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "layer" << i+1 << "'s leakagePower is: " << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << "uW" << endl;
			layerfile << numTileEachLayer[0][i] * numTileEachLayer[1][i] * tileLeakage*1e6 << ",";
			cout << "layer" << i+1 << "'s leakageEnergy is: " << layerLeakageEnergy*1e12 << "pJ" << endl;
			layerfile << layerLeakageEnergy*1e12 << ",";

			cout << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADC*1e9 << "ns" << endl;
			layerfile << coreLatencyADC*1e9 << ",";
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccum*1e9 << "ns" << endl;
			layerfile << coreLatencyAccum*1e9 << ",";
			cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readLatency is : " << coreLatencyOther*1e9 << "ns" << endl;
			layerfile << coreLatencyOther*1e9 << ",";
			cout << "----------- Buffer buffer latency is: " << layerbufferLatency*1e9 << "ns" << endl;
			layerfile << layerbufferLatency*1e9 << ",";
			cout << "----------- Interconnect latency is: " << layericLatency*1e9 << "ns" << endl;
			layerfile << layericLatency*1e9 << ",";
			cout << "----------- Weight Gradient Calculation readLatency is : " << layerReadLatencyPeakWG*1e9 << "ns" << endl;
			layerfile << layerReadLatencyPeakWG*1e9 << ",";
			cout << "----------- Weight Update writeLatency is : " << layerWriteLatencyPeakWU*1e9 << "ns" << endl;
			layerfile << layerWriteLatencyPeakWU*1e9 << ",";
			cout << "----------- DRAM data transfer Latency is : " << layerDRAMLatency*1e9 << "ns" << endl;
			layerfile << layerDRAMLatency*1e9 << ",";
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADC*scalingFactor_Total*1e12 << "pJ" << endl;
			layerfile << coreEnergyADC*scalingFactor_Total*1e12 << ",";
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccum*scalingFactor_Total*1e12 << "pJ" << endl;
			layerfile << coreEnergyAccum*scalingFactor_Total*1e12 << ",";
			cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readDynamicEnergy is : " << coreEnergyOther*1e12 << "pJ" << endl;
			layerfile << coreEnergyOther*1e12 << ",";
			cout << "----------- Buffer readDynamicEnergy is: " << layerbufferDynamicEnergy*1e12 << "pJ" << endl;
			layerfile << layerbufferDynamicEnergy*1e12 << ",";
			cout << "----------- Interconnect readDynamicEnergy is: " << layericDynamicEnergy*1e12 << "pJ" << endl;
			layerfile << layericDynamicEnergy*1e12 << ",";
			cout << "----------- Weight Gradient Calculation readDynamicEnergy is : " << layerReadDynamicEnergyPeakWG*scalingFactor_Total*1e12 << "pJ" << endl;
			layerfile << layerReadDynamicEnergyPeakWG*scalingFactor_Total*1e12 << ",";
			cout << "----------- Weight Update writeDynamicEnergy is : " << layerWriteDynamicEnergyPeakWU*1e12 << "pJ" << endl;
			layerfile << layerWriteDynamicEnergyPeakWU*1e12 << ",";
			cout << "----------- DRAM data transfer Energy is : " << layerDRAMDynamicEnergy*1e12 << "pJ" << endl;
			layerfile << layerDRAMDynamicEnergy*1e12 << endl;
			cout << endl;

			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;

            // My addition
            if (param->rule == "dfa" && chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU < layerReadLatencyAG+layerReadLatencyWG+layerWriteLatencyWU) {
                chipReadLatencyAG = layerReadLatencyAG;
                chipReadLatencyWG = layerReadLatencyWG;
                chipWriteLatencyWU = layerWriteLatencyWU;
                chipReadLatencyPeakAG = layerReadLatencyPeakAG;
                chipReadLatencyPeakWG = layerReadLatencyPeakWG;
                chipWriteLatencyPeakWU = layerWriteLatencyPeakWU;
            }
            else {
			    chipReadLatencyAG += layerReadLatencyAG;
                chipReadLatencyWG += layerReadLatencyWG;
                chipWriteLatencyWU += layerWriteLatencyWU;
                chipReadLatencyPeakAG += layerReadLatencyPeakAG;
                chipReadLatencyPeakWG += layerReadLatencyPeakWG;
                chipWriteLatencyPeakWU += layerWriteLatencyPeakWU;
            }

			chipReadLatency += layerReadLatency;
			chipReadDynamicEnergy += layerReadDynamicEnergy;
			chipReadDynamicEnergyAG += layerReadDynamicEnergyAG;
			chipReadDynamicEnergyWG += layerReadDynamicEnergyWG;
			chipWriteDynamicEnergyWU += layerWriteDynamicEnergyWU;
			chipDRAMLatency += layerDRAMLatency;
			chipDRAMDynamicEnergy += layerDRAMDynamicEnergy;

			chipReadLatencyPeakFW += layerReadLatencyPeakFW;
			chipReadDynamicEnergyPeakFW += layerReadDynamicEnergyPeakFW;
			chipReadDynamicEnergyPeakAG += layerReadDynamicEnergyPeakAG;
			chipReadDynamicEnergyPeakWG += layerReadDynamicEnergyPeakWG;
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

            param->activityRowReadWG = atof(argv[4*i+17]);
            param->activityRowWriteWG = atof(argv[4*i+17]);
            param->activityColWriteWG = atof(argv[4*i+17]);
			ChipCalculatePerformance(inputParameter, tech, cell, i, argv[4*i+14], argv[4*i+15], argv[4*i+16], netStructure[i][6],
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
        layerfile << "# of Tiles, Speed-up, Utilization, Read Latency (ns), Read Dynamic Energy (pJ), Read Latency of Activation Gradient (ns), Read Dynamic Energy of Activation Gradient (pJ), " <<
        "Read Latency of Weight Gradient (ns), Read Dynamic Energy of Weight Gradient (pJ), Write Latency of Weight Update (ns), Write Dynamic Energy of Weight Update (pJ), " <<
        "PEAK Read Latency (ns), PEAK Read Dynamic Energy (pJ), PEAK Read Latency of Activation Gradient (ns), PEAK Read Dynamic Energy of Activation Gradient (pJ), " <<
        "PEAK Read Latency of Weight Gradient (ns), PEAK Read Dynamic Energy of Weight Gradient (pJ), PEAK writeLatency of Weight Update (ns), " <<
        "PEAK writeDynamicEnergy of Weight Update (pJ), Leakage Power (uW), Leakage Energy (pJ), ADC Read Latency (ns), Accumulation Circuits Read Latency (ns), " <<
        "Synaptic Array w/o ADC Read Latency (ns), Buffer Latency (ns), Interconnect Latency (ns), Weight Gradient Calculation Read Latency (ns), " <<
        "Weight Update Write Latency (ns), DRAM data transfer Latency (ns), ADC Read Dynamic Energy (pJ), Accumulation Circuits Read Dynamic Energy (pJ), " <<
        "Synaptic Array w/o ADC Read Dynamic Energy (pJ), Buffer Read Dynamic Energy (pJ), Interconnect Read Dynamic Energy (pJ), Weight Gradient Calculation Read Dynamic Energy (pJ), " <<
        "Weight Update Write Dynamic Energy (pJ), DRAM data transfer Dynamic Energy (pJ)" << endl;
		for (int i=0; i<netStructure.size(); i++) {
            // Build layer estimation csv file, one row for each layer
			cout << "-------------------- Estimation of Layer " << i+1 << " ----------------------" << endl;

            layerfile << numTileEachLayer[0][i] * numTileEachLayer[1][i] << ", ";
            layerfile << speedUpEachLayer[0][i] * speedUpEachLayer[1][i] << ", ";
            layerfile << utilizationEachLayer[i][0] << ", ";
			cout << "layer" << i+1 << "'s readLatency is: " << readLatencyPerLayer[i]*1e9 << "ns" << endl;
			layerfile << readLatencyPerLayer[i]*1e9 << ",";
			cout << "layer" << i+1 << "'s readDynamicEnergy is: " << readDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
			layerfile << readDynamicEnergyPerLayer[i]*1e12 << ",";
			cout << "layer" << i+1 << "'s readLatency of Activation Gradient is: " << readLatencyPerLayerAG[i]*1e9 << "ns" << endl;
			layerfile << readLatencyPerLayerAG[i]*1e9 << ",";
			cout << "layer" << i+1 << "'s readDynamicEnergy of Activation Gradient is: " << readDynamicEnergyPerLayerAG[i]*1e12 << "pJ" << endl;
			layerfile << readDynamicEnergyPerLayerAG[i]*1e12 << ",";
			cout << "layer" << i+1 << "'s readLatency of Weight Gradient is: " << readLatencyPerLayerWG[i]*1e9 << "ns" << endl;
			layerfile << readLatencyPerLayerWG[i]*1e9 << ",";
			cout << "layer" << i+1 << "'s readDynamicEnergy of Weight Gradient is: " << readDynamicEnergyPerLayerWG[i]*1e12 << "pJ" << endl;
			layerfile << readDynamicEnergyPerLayerWG[i]*1e12 << ",";
			cout << "layer" << i+1 << "'s writeLatency of Weight Update is: " << writeLatencyPerLayerWU[i]*1e9 << "ns" << endl;
			layerfile << writeLatencyPerLayerWU[i]*1e9 << ",";
			cout << "layer" << i+1 << "'s writeDynamicEnergy of Weight Update is: " << writeDynamicEnergyPerLayerWU[i]*1e12 << "pJ" << endl;
			layerfile << writeDynamicEnergyPerLayerWU[i]*1e12 << ",";
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "layer" << i+1 << "'s PEAK readLatency is: " << readLatencyPerLayerPeakFW[i]*1e9 << "ns" << endl;
			layerfile << readLatencyPerLayerPeakFW[i]*1e9 << ",";
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy is: " << readDynamicEnergyPerLayerPeakFW[i]*1e12 << "pJ" << endl;
			layerfile << readDynamicEnergyPerLayerPeakFW[i]*1e12 << ",";
			cout << "layer" << i+1 << "'s PEAK readLatency of Activation Gradient is: " << readLatencyPerLayerPeakAG[i]*1e9 << "ns" << endl;
			layerfile << readLatencyPerLayerPeakAG[i]*1e9 << ",";
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Activation Gradient is: " << readDynamicEnergyPerLayerPeakAG[i]*1e12 << "pJ" << endl;
			layerfile << readDynamicEnergyPerLayerPeakAG[i]*1e12 << ",";
			cout << "layer" << i+1 << "'s PEAK readLatency of Weight Gradient is: " << readLatencyPerLayerPeakWG[i]*1e9 << "ns" << endl;
			layerfile << readLatencyPerLayerPeakWG[i]*1e9 << ",";
			cout << "layer" << i+1 << "'s PEAK readDynamicEnergy of Weight Gradient is: " << readDynamicEnergyPerLayerPeakWG[i]*1e12 << "pJ" << endl;
			layerfile << readDynamicEnergyPerLayerPeakWG[i]*1e12 << ",";
			cout << "layer" << i+1 << "'s PEAK writeLatency of Weight Update is: " << writeLatencyPerLayerPeakWU[i]*1e9 << "ns" << endl;
			layerfile << writeLatencyPerLayerPeakWU[i]*1e9 << ",";
			cout << "layer" << i+1 << "'s PEAK writeDynamicEnergy of Weight Update is: " << writeDynamicEnergyPerLayerPeakWU[i]*1e12 << "pJ" << endl;
			layerfile << writeDynamicEnergyPerLayerPeakWU[i]*1e12 << ",";
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "layer" << i+1 << "'s leakagePower is: " << leakagePowerPerLayer[i]*1e6 << "uW" << endl;
			layerfile << leakagePowerPerLayer[i]*1e6 << ",";
			cout << "layer" << i+1 << "'s leakageEnergy is: " << leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]) *1e12 << "pJ" << endl;
			layerfile << leakagePowerPerLayer[i] * (systemClock-readLatencyPerLayer[i]) *1e12 << ",";
			cout << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << coreLatencyADCPerLayer[i]*1e9 << "ns" << endl;
			layerfile << coreLatencyADCPerLayer[i]*1e9 << ",";
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << coreLatencyAccumPerLayer[i]*1e9 << "ns" << endl;
			layerfile << coreLatencyAccumPerLayer[i]*1e9 << ",";
			cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readLatency is : " << coreLatencyOtherPerLayer[i]*1e9 << "ns" << endl;
			layerfile << coreLatencyOtherPerLayer[i]*1e9 << ",";
			cout << "----------- Buffer latency is: " << bufferLatencyPerLayer[i]*1e9 << "ns" << endl;
			layerfile << bufferLatencyPerLayer[i]*1e9 << ",";
			cout << "----------- Interconnect latency is: " << icLatencyPerLayer[i]*1e9 << "ns" << endl;
			layerfile << icLatencyPerLayer[i]*1e9 << ",";
			cout << "----------- Weight Gradient Calculation readLatency is : " << readLatencyPerLayerPeakWG[i]*1e9 << "ns" << endl;
			layerfile << readLatencyPerLayerPeakWG[i]*1e9 << ",";
			cout << "----------- Weight Update writeLatency is : " << writeLatencyPerLayerPeakWU[i]*1e9 << "ns" << endl;
			layerfile << writeLatencyPerLayerPeakWU[i]*1e9 << ",";
			cout << "----------- DRAM data transfer Latency is : " << dramLatencyPerLayer[i]*1e9 << "ns" << endl;
			layerfile << dramLatencyPerLayer[i]*1e9 << ",";
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << coreEnergyADCPerLayer[i]*1e12 << "pJ" << endl;
			layerfile << coreEnergyADCPerLayer[i]*1e12 << ",";
			cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << coreEnergyAccumPerLayer[i]*1e12 << "pJ" << endl;
			layerfile << coreEnergyAccumPerLayer[i]*1e12 << ",";
			cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readDynamicEnergy is : " << coreEnergyOtherPerLayer[i]*1e12 << "pJ" << endl;
			layerfile << coreEnergyOtherPerLayer[i]*1e12 << ",";
			cout << "----------- Buffer readDynamicEnergy is: " << bufferEnergyPerLayer[i]*1e12 << "pJ" << endl;
			layerfile << bufferEnergyPerLayer[i]*1e12 << ",";
			cout << "----------- Interconnect readDynamicEnergy is: " << icEnergyPerLayer[i]*1e12 << "pJ" << endl;
			layerfile << icEnergyPerLayer[i]*1e12 << ",";
			cout << "----------- Weight Gradient Calculation readDynamicEnergy is : " << readDynamicEnergyPerLayerPeakWG[i]*1e12 << "pJ" << endl;
			layerfile << readDynamicEnergyPerLayerPeakWG[i]*1e12 << ",";
			cout << "----------- Weight Update writeDynamicEnergy is : " << writeDynamicEnergyPerLayerPeakWU[i]*1e12 << "pJ" << endl;
			layerfile << writeDynamicEnergyPerLayerPeakWU[i]*1e12 << ",";
			cout << "----------- DRAM data transfer DynamicEnergy is : " << dramDynamicEnergyPerLayer[i]*1e12 << "pJ" << endl;
			layerfile << dramDynamicEnergyPerLayer[i]*1e12 << endl;
			cout << endl;
			cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
			cout << endl;

			chipLeakageEnergy += leakagePowerPerLayer[i] * ((systemClock-readLatencyPerLayer[i]) + (systemClockAG-readLatencyPerLayerAG[i]));

		}
		layerfile.close();
	}

    ofstream summaryfile;
    summaryfile.open("Summary.csv", ios::out);
    summaryfile << "Memory Utilization (%), Chip Area (um^2), Chip total CIM array (um^2), Total IC Area on chip (um^2), Total ADC Area on chip (um^2), Total Accumulation Circuits on chip (um^2), " <<
      "Other Peripheries (um^2), Weight Gradient Calculation (um^2), Chip Read Latency of Forward (ns), Chip Read Dynamic Energy of Forward (pJ), " <<
      "Chip Read Latency of Activation Gradient (ns), Chip Read Dynamic Energy of Activation Gradient (pJ), Chip Read Latency of Weight Gradient (ns), " <<
      "Chip Read Dynamic Energy of Weight Gradient (pJ), Chip Write Latency of Weight Update (ns), Chip Write Dynamic Energy of Weight Update (pJ), " <<
      "Chip total Latency (ns), Chip total Energy (pJ), Chip PEAK Read Latency of Forward (ns), Chip PEAK Read Dynamic Energy of Forward (ns), " <<
      "Chip PEAK Read Latency of Activation Gradient (ns), Chip PEAK Read Dynamic Energy of Activation Gradient (pJ), Chip PEAK Read Latency of Weight Gradient (ns), " <<
      "Chip PEAK Read Dynamic Energy of Weight Gradient (pJ), Chip PEAK writeLatency of Weight Update (ns), Chip PEAK writeDynamicEnergy of Weight Update (pJ), " <<
      "Chip PEAK total Latency (ns), Chip PEAK total Energy (pJ), Chip leakage Energy (pJ), Chip leakage Power (uW), ADC Read Latency (ns), " <<
      "Accumulation Circuits Read Latency (ns), Synaptic Array w/o ADC Read Latency  (ns), Buffer Read Latency (ns), Interconnect Read Latency (ns), " <<
      "Weight Gradient Calculation Read Latency (ns), Weight Update Write Latency (ns), DRAM data transfer Latency (ns), ADC Read Dynamic Energy (pJ), " <<
      "Accumulation Circuits Read Dynamic Energy (pJ), Synaptic Array w/o ADC Read Dynamic Energy (pJ), Buffer Read Dynamic Energy (pJ), " <<
      "Interconnect Read Dynamic Energy (pJ), Weight Gradient Calculation Read Dynamic Energy (pJ), Weight Update Write Dynamic Energy (pJ), " <<
      "DRAM data transfer Dynamic Energy (pJ), Energy Efficiency TOPS/W, Throughput TOPS, Throughput FPS, Peak Energy Efficiency TOPS/W, " <<
      "Peak Throughput TOPS, Peak Throughput FPS" << endl;

    summaryfile << realMappedMemory/totalNumTile*100 << ", ";
	cout << "------------------------------ Summary --------------------------------" <<  endl;
	cout << endl;
	cout << "ChipArea : " << chipArea*1e12 << "um^2" << endl;
	summaryfile << chipArea*1e12 << ",";
	cout << "Chip total CIM (Forward+Activation Gradient) array : " << chipAreaArray*1e12 << "um^2" << endl;
	summaryfile << chipAreaArray*1e12 << ",";
	cout << "Total IC Area on chip (Global and Tile/PE local): " << chipAreaIC*1e12 << "um^2" << endl;
	summaryfile << chipAreaIC*1e12 << ",";
	cout << "Total ADC (or S/As and precharger for SRAM) Area on chip : " << chipAreaADC*1e12 << "um^2" << endl;
	summaryfile << chipAreaADC*1e12 << ",";
	cout << "Total Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) on chip : " << chipAreaAccum*1e12 << "um^2" << endl;
	summaryfile << chipAreaAccum*1e12 << ",";
	cout << "Other Peripheries (e.g. decoders, mux, switchmatrix, buffers, pooling and activation units) : " << chipAreaOther*1e12 << "um^2" << endl;
	summaryfile << chipAreaOther*1e12 << ",";
	cout << "Weight Gradient Calculation : " << chipAreaWG*1e12 << "um^2" << endl;
	summaryfile << chipAreaWG*1e12 << ",";
	cout << endl;
	if (! param->pipeline) {
		cout << "-----------------------------------Chip layer-by-layer Estimation---------------------------------" << endl;
	} else {
		cout << "--------------------------------------Chip pipeline Estimation---------------------------------" << endl;
	}
	cout << "Chip readLatency of Forward (per epoch) is: " << chipReadLatency*1e9 << "ns" << endl;
	summaryfile << chipReadLatency*1e9 << ",";
	cout << "Chip readDynamicEnergy of Forward (per epoch) is: " << chipReadDynamicEnergy*1e12 << "pJ" << endl;
	summaryfile << chipReadDynamicEnergy*1e12 << ",";
	cout << "Chip readLatency of Activation Gradient (per epoch) is: " << chipReadLatencyAG*1e9 << "ns" << endl;
	summaryfile << chipReadLatencyAG*1e9 << ",";
	cout << "Chip readDynamicEnergy of Activation Gradient (per epoch) is: " << chipReadDynamicEnergyAG*1e12 << "pJ" << endl;
	summaryfile << chipReadDynamicEnergyAG*1e12 << ",";
	cout << "Chip readLatency of Weight Gradient (per epoch) is: " << chipReadLatencyWG*1e9 << "ns" << endl;
	summaryfile << chipReadLatencyWG*1e9 << ",";
	// My addition
	cout << "Chip readDynamicEnergy of Weight Gradient (per epoch) is: " << chipReadDynamicEnergyWG*scalingFactor_WG*1e12 << "pJ" << endl;
	summaryfile << chipReadDynamicEnergyWG*scalingFactor_WG*1e12 << ",";
	cout << "Chip writeLatency of Weight Update (per epoch) is: " << chipWriteLatencyWU*1e9 << "ns" << endl;
	summaryfile << chipWriteLatencyWU*1e9 << ",";
	cout << "Chip writeDynamicEnergy of Weight Update (per epoch) is: " << chipWriteDynamicEnergyWU*1e12 << "pJ" << endl;
	summaryfile << chipWriteDynamicEnergyWU*1e12 << ",";
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "Chip total Latency (per epoch) is: " << (chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e9 << "ns" << endl;
	summaryfile << (chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e9 << ",";
	// My addition
	cout << chipReadDynamicEnergy << endl;
	cout << chipReadDynamicEnergyAG << endl;
	cout << chipReadDynamicEnergyWG << endl;
	cout << chipWriteDynamicEnergyWU << endl;

	cout << "Chip total Energy (per epoch) is: " << (chipReadDynamicEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*scalingFactor_Total*1e12 << "pJ" << endl;
	summaryfile << (chipReadDynamicEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*scalingFactor_Total*1e12 << ",";
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "Chip PEAK readLatency of Forward (per epoch) is: " << chipReadLatencyPeakFW*1e9 << "ns" << endl;
	summaryfile << chipReadLatencyPeakFW*1e9 << ",";
	cout << "Chip PEAK readDynamicEnergy of Forward (per epoch) is: " << chipReadDynamicEnergyPeakFW*1e12 << "pJ" << endl;
	summaryfile << chipReadDynamicEnergyPeakFW*1e12 << ",";
	cout << "Chip PEAK readLatency of Activation Gradient (per epoch) is: " << chipReadLatencyPeakAG*1e9 << "ns" << endl;
	summaryfile << chipReadLatencyPeakAG*1e9 << ",";
	cout << "Chip PEAK readDynamicEnergy of Activation Gradient (per epoch) is: " << chipReadDynamicEnergyPeakAG*1e12 << "pJ" << endl;
	summaryfile << chipReadDynamicEnergyPeakAG*1e12 << ",";
	cout << "Chip PEAK readLatency of Weight Gradient (per epoch) is: " << chipReadLatencyPeakWG*1e9 << "ns" << endl;
	summaryfile << chipReadLatencyPeakWG*1e9 << ",";
	// My addition
	cout << "Chip PEAK readDynamicEnergy of Weight Gradient (per epoch) is: " << chipReadDynamicEnergyPeakWG*scalingFactor_WG*1e12 << "pJ" << endl;
	summaryfile << chipReadDynamicEnergyPeakWG*scalingFactor_WG*1e12 << ",";
	cout << "Chip PEAK writeLatency of Weight Update (per epoch) is: " << chipWriteLatencyPeakWU*1e9 << "ns" << endl;
	summaryfile << chipWriteLatencyPeakWU*1e9 << ",";
	cout << "Chip PEAK writeDynamicEnergy of Weight Update (per epoch) is: " << chipWriteDynamicEnergyPeakWU*1e12 << "pJ" << endl;
	summaryfile << chipWriteDynamicEnergyPeakWU*1e12 << ",";
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "Chip PEAK total Latency (per epoch) is: " << (chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e9 << "ns" << endl;
	summaryfile << (chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e9 << ",";
	// My addition
	cout << "Chip PEAK total Energy (per epoch) is: " << (chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*scalingFactor_Total*1e12 << "pJ" << endl;
	summaryfile << (chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*scalingFactor_Total*1e12 << ",";
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "Chip leakage Energy is: " << chipLeakageEnergy*1e12 << "pJ" << endl;
	summaryfile << chipLeakageEnergy*1e12 << ",";
	cout << "Chip leakage Power is: " << chipLeakage*1e6 << "uW" << endl;
	summaryfile << chipLeakage*1e6 << ",";
	cout << endl;
	cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
	cout << endl;
	cout << "----------- ADC (or S/As and precharger for SRAM) readLatency is : " << chipLatencyADC*1e9 << "ns" << endl;
	summaryfile << chipLatencyADC*1e9 << ",";
	// My addition
	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readLatency is : " << chipLatencyAccum*scalingFactor_Total*1e9 << "ns" << endl;
	summaryfile << chipLatencyAccum*scalingFactor_Total*1e9 << ",";
	cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readLatency is : " << chipLatencyOther*1e9 << "ns" << endl;
	summaryfile << chipLatencyOther*1e9 << ",";
	cout << "----------- Buffer readLatency is: " << chipbufferLatency*1e9 << "ns" << endl;
	summaryfile << chipbufferLatency*1e9 << ",";
	cout << "----------- Interconnect readLatency is: " << chipicLatency*1e9 << "ns" << endl;
	summaryfile << chipicLatency*1e9 << ",";
	cout << "----------- Weight Gradient Calculation readLatency is : " << chipReadLatencyPeakWG*1e9 << "ns" << endl;
	summaryfile << chipReadLatencyPeakWG*1e9 << ",";
	cout << "----------- Weight Update writeLatency is : " << chipWriteLatencyPeakWU*1e9 << "ns" << endl;
	summaryfile << chipWriteLatencyPeakWU*1e9 << ",";
	cout << "----------- DRAM data transfer Latency is : " << chipDRAMLatency*1e9 << "ns" << endl;
	summaryfile << chipDRAMLatency*1e9 << ",";
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	// My addition
	cout << "----------- ADC (or S/As and precharger for SRAM) readDynamicEnergy is : " << chipEnergyADC*scalingFactor_Total*1e12 << "pJ" << endl;
	summaryfile << chipEnergyADC*scalingFactor_Total*1e12 << ",";
	// My addition
	cout << "----------- Accumulation Circuits (subarray level: adders, shiftAdds; PE/Tile/Global level: accumulation units) readDynamicEnergy is : " << chipEnergyAccum*scalingFactor_Total*1e12 << "pJ" << endl;
	summaryfile << chipEnergyAccum*scalingFactor_Total*1e12 << ",";
	cout << "----------- Synaptic Array w/o ADC (Forward + Activate Gradient) readDynamicEnergy is : " << chipEnergyOther*1e12 << "pJ" << endl;
	summaryfile << chipEnergyOther*1e12 << ",";
	cout << "----------- Buffer readDynamicEnergy is: " << chipbufferReadDynamicEnergy*1e12 << "pJ" << endl;
	summaryfile << chipbufferReadDynamicEnergy*1e12 << ",";
	cout << "----------- Interconnect readDynamicEnergy is: " << chipicReadDynamicEnergy*1e12 << "pJ" << endl;
	summaryfile << chipicReadDynamicEnergy*1e12 << ",";
	cout << "----------- Weight Gradient Calculation readDynamicEnergy is : " << chipReadDynamicEnergyPeakWG*scalingFactor_WG*1e12 << "pJ" << endl;
	summaryfile << chipReadDynamicEnergyPeakWG*scalingFactor_WG*1e12 << ",";
	cout << "----------- Weight Update writeDynamicEnergy is : " << chipWriteDynamicEnergyPeakWU*1e12 << "pJ" << endl;
	summaryfile << chipWriteDynamicEnergyPeakWU*1e12 << ",";
	cout << "----------- DRAM data transfer DynamicEnergy is : " << chipDRAMDynamicEnergy*1e12 << "pJ" << endl;
	summaryfile << chipDRAMDynamicEnergy*1e12 << ",";
	cout << endl;
	cout << "************************ Breakdown of Latency and Dynamic Energy *************************" << endl;
	cout << endl;
	cout << endl;
	if (! param->pipeline) {
		cout << "-----------------------------------Chip layer-by-layer Performance---------------------------------" << endl;
	} else {
		cout << "--------------------------------------Chip pipeline Performance---------------------------------" << endl;
	}

    // My addition
	cout << "Energy Efficiency TOPS/W: " << numComputation/((chipReadDynamicEnergy+chipLeakageEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*scalingFactor_Total*1e12) << endl;
	summaryfile << numComputation/((chipReadDynamicEnergy+chipLeakageEnergy+chipReadDynamicEnergyAG+chipReadDynamicEnergyWG+chipWriteDynamicEnergyWU)*scalingFactor_Total*1e12) << ",";
	cout << "Throughput TOPS: " << numComputation/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e-12 << endl;
	summaryfile << numComputation/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU)*1e-12 << ",";
	cout << "Throughput FPS: " << 1/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU) << endl;
	summaryfile << 1/(chipReadLatency+chipReadLatencyAG+chipReadLatencyWG+chipWriteLatencyWU) << ",";
	cout << "--------------------------------------------------------------------------" << endl;
	// My addition
	cout << "Peak Energy Efficiency TOPS/W: " << numComputation/((chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*scalingFactor_Total*1e12) << endl;
	summaryfile << numComputation/((chipReadDynamicEnergyPeakFW+chipReadDynamicEnergyPeakAG+chipReadDynamicEnergyPeakWG+chipWriteDynamicEnergyPeakWU)*scalingFactor_Total*1e12) << ",";
	cout << "Peak Throughput TOPS: " << numComputation/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e-12 << endl;
	summaryfile << numComputation/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU)*1e-12 << ",";
	cout << "Peak Throughput FPS: " << 1/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU) << endl;
    summaryfile << 1/(chipReadLatencyPeakFW+chipReadLatencyPeakAG+chipReadLatencyPeakWG+chipWriteLatencyPeakWU) << endl;

	cout << "-------------------------------------- Hardware Performance Done --------------------------------------" <<  endl;
	cout << endl;
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