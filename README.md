# M.Sc. Thesis

## LLM-based Multilingual Information Extraction and Classification for Antimicrobial Resistance Policy Documents

This repository contains the code and experimental implementations developed as part of my M.Sc. Artificial Intelligence thesis at City St George's, University of London.

The thesis investigates computational approaches for extracting and classifying information from antimicrobial resistance policy documents. The work explores statistical, machine learning, retrieval augmented generation, and large language model based approaches for transforming unstructured policy text into structured representations using predefined policy coding dimensions.

The research was conducted as part of the SPAARTA project, in collaboration with the SPAARTA team at Imperial College London through the Fleming Initiative, and in alignment with the Health Policy Sciences team at City St George's, University of London.

## Experiments

The repository contains implementations and experimental code for the following approaches:

- Statistical baseline methods, including frequency based priors and k nearest neighbours
- Large language model based intervention summarisation
- Vanilla retrieval augmented generation
- Modern hybrid retrieval augmented generation
- Quote Then Judge classification using fine tuned language models
- Five channel fusion combining statistical, semantic, categorical, and definition based signals

The experiments evaluate the ability of these approaches to identify and classify antimicrobial resistance policy interventions across multiple coding dimensions.

## Repository Structure

The repository is organised according to the experiments described in the thesis. Each experiment contains the relevant implementation and supporting code used during development and evaluation.

The main experimental components include baseline classification, retrieval augmented generation, language model based classification, and multi channel fusion approaches.

## Dataset

The experiments use a collection of antimicrobial resistance policy documents and corresponding expert coded interventions provided through the research collaboration.

Due to confidentiality and data sharing restrictions, the underlying policy documents and human coded dataset are not included in this repository.

The repository therefore contains the experimental implementations and supporting code required to document the computational approaches described in the thesis, while the original research data remains within the appropriate research environment.

## Reproducibility

The code in this repository corresponds to the experimental methods and configurations described in the thesis. Model configurations, retrieval approaches, classification procedures, and evaluation methods are retained alongside the respective experiments where applicable.

Because the underlying dataset is confidential and some experiments require specific computational resources and pretrained models, complete reproduction of every experiment may require access to the original research environment and data.

## Thesis

This repository accompanies the M.Sc. Artificial Intelligence thesis:

**LLM-based Multilingual Information Extraction and Classification for Antimicrobial Resistance Policy Documents**

**Author:** Sania Verma  
**Programme:** M.Sc. Artificial Intelligence  
**Institution:** City St George's, University of London  
**Supervisor:** Dr Tillman Weyde

## Code Availability

The code and experimental implementations developed for this research are provided in this repository for transparency and reproducibility. The underlying policy documents and expert coded dataset are not publicly distributed due to confidentiality restrictions associated with the research collaboration.

The access to the fine tuned models and their checkpoints is provided here in the drive link : (https://drive.google.com/drive/folders/1E9fF1dMfAj9m22zxjVYxz0F-YN6kOmtG?usp=sharing). 
