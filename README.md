# OODBench

Welcome to the OOD Bench repository!

## Project Scope 🔍
OOD Bench is a research project aimed at assesing the behaviour of model Decision Boundaries(DB) when classifying out-of-distribution data. The project evaluates various models and techniques for identifying and handling out-of-distribution samples in machine learning and computer vision applications. The experiments and DB visualization are based on  [Somepalli's paper repository ](https://drive.google.com/file/d/1Wa8tBWWWK3QZuKe1GADxhvniGDOPldNy/view?usp=sharing). This porject is framed in the course Tutored Research and Development Project(TRDP) of the international Image Processing and Computer Vision MSc. 

## Research Report 📑
For detailed insights and results of the first part of the research, please refer to our research report:
- [Research Report](https://drive.google.com/file/d/1Wa8tBWWWK3QZuKe1GADxhvniGDOPldNy/view?usp=sharing).

The results obtained in the original experiment (the one reported) can be recreated by running:

```
$ python extract_margin_accuracy_triplets_legacy.py
```

Feel free to explore the code, datasets, and resources in this repository to learn more about our research and findings.

## Getting Started 👣
To get started with OOD Bench, follow these steps:

1. Clone this repository to your local machine.
2. Download the default weights from the provided link (if needed) and save then in the "pretrained_models" folder.
3. Explore the code, datasets, and notebooks to replicate experiments or conduct your own research.

## Weights
By default, this project utilizes the following weights for ResNet18:
- [resnet18-5c106cde.pth](https://drive.google.com/file/d/1aiHE_pdOsiX0_gVrZhZAB8gNbOpC_iHq/view?usp=sharing)
Other weights are avaliable [in this folder](https://drive.google.com/drive/folders/1xPCseAqVNC7R9D5cfkSwe_mJWivzOFwM?usp=sharing).

## OODatasets
The developed method uses a series of datasets that are contained in the following zip:[OODatasets.zip](https://drive.google.com/file/d/1L-1qM3Gkod0Vw2JlwfDEsrFukCSqCLxi/view?usp=sharing)
These folders contain a subset of images for 3 classes that were present in all the OOD datasets:

- "n02106662"  German shepard
- "n03388043"  Fountain
- "n03594945"  Jeep

To-do: add references of datasets.

### Authors: Blanca Hermosilla and Juan Manuel Peña
### Proponents: Marcos Escudero and Pierre Jacob


If you have any questions, suggestions, or feedback, please don't hesitate to reach out.

Happy OOD researching!
