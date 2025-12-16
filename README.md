# Federated-Learning-Context-Aware

## Overview
This repository contains the first draft of the code for the experiments presented in our paper:

**Palomares, Javier, Chiara Camerota, Estefanía Coronado, Cristina Cervelló-Pastor, Muhammad Shuaib Siddiqui, and Flavio Esposito. "Taming Bandwidth Bottlenecks in Federated Learning via ECN-based Gradient Compression."**

The work focuses on improving **federated learning** efficiency under constrained network bandwidth by using **ECN-based gradient compression** techniques. The repository supports context-aware federated learning experiments across multiple datasets and network conditions.

## Research Context
Federated learning enables collaborative model training without sharing raw data, but bandwidth bottlenecks often limit scalability. This project addresses **network congestion and communication efficiency** by integrating **ECN-aware gradient compression**. It is relevant to IoT, edge AI, and distributed systems where network constraints are critical.

## Methodology
- **Models:** Standard neural networks (e.g., CNNs, MLPs) trained in a federated manner  
- **Compression:** ECN-based gradient compression to reduce communication load  
- **Data:** Multiple datasets simulated in federated environments  
- **Evaluation:** Performance measured in terms of model accuracy, communication efficiency, and bandwidth usage

## System Architecture
```text
            ┌──────────────────────┐
            │   Central Server     │
            └─────────┬────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
   ┌───────────────┐       ┌───────────────┐
   │ Client Node 1 │       │ Client Node N │
   └───────────────┘       └───────────────┘
          │                       │
          └─────ECN Gradient──────┘
                Compression

```

# Results

The experiments demonstrate that ECN-based gradient compression can significantly reduce communication overhead without sacrificing model accuracy. Detailed results, plots, and metrics are included in the results/ folder of this repository. 

# References

Palomares, Javier, Chiara Camerota, Estefanía Coronado, Cristina Cervelló-Pastor, Muhammad Shuaib Siddiqui, and Flavio Esposito. "Taming Bandwidth Bottlenecks in Federated Learning via ECN-based Gradient Compression."
