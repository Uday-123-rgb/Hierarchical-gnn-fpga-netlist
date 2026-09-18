# Hierarchical-gnn-fpga-netlist
Hierarchical GNN-based classification of digital circuit architectures from synthesized FPGA netlists.
# Hierarchical GNN for FPGA Netlist Classification

This repository contains the source code and trained Graph Neural Network (GNN) models developed for the classification of digital circuit architectures from synthesized FPGA netlists.

## Project Overview

The project uses a Hierarchical Graph Neural Network to analyze synthesized FPGA netlists and identify the underlying digital circuit architecture.

The overall workflow is:

Parameterized RTL
        ↓
FPGA Synthesis
        ↓
EDIF / EDN Netlist
        ↓
Netlist Parsing
        ↓
Graph Construction
        ↓
Feature Extraction
        ↓
GNN Processing
        ↓
Hierarchical Classification

The framework considers circuit families including:

- Shift Registers
- Registers / Storage
- Counters
- Combinational Logic
- Finite-State Machines (FSMs)

## Repository Contents

### 📁 Code

The `code/` folder contains the implementation developed for the project, including:

- Netlist parsing
- Graph construction
- Node feature extraction
- Metadata extraction
- GNN architecture
- Model training
- Model evaluation
- Supporting scripts and notebooks

### 📁 Trained Models

The `trained_models/` folder contains the trained GNN model files used for circuit architecture classification.

The models correspond to the hierarchical classification framework and its family-specific classifiers.

## Technologies Used

- Python
- PyTorch
- PyTorch Geometric
- Verilog
- Xilinx Vivado
- Jupyter Notebook

## Model Architecture

The GNN framework uses Graph Isomorphism Network (GIN) layers to learn structural information from FPGA netlist graphs.

The graph representation consists of:

- FPGA primitive nodes
- Signal connectivity edges
- Primitive-type features
- Structural attributes
- LUT INIT-derived information

The learned graph representation is used for graph-level circuit architecture classification.

## Applications

This work can support:

- FPGA netlist analysis
- Digital circuit identification
- Hardware reverse engineering
- Hardware security research
- Architectural information recovery from synthesized netlists

## Authors

**Uday Vaidya**  
**Yash Bendkar**  
**Nanditha N Varma**  
**Dr. Jayaraj U Kidav**

Department of Electronics Engineering  
NIELIT, India

## Status

Research / Development
