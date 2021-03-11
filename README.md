# Setup
**Requirements:**
- NVIDIA GPU
- Ubuntu 18.04
- Python >=3.5

**Python packages**
sklearn
numpy
pytorch 1.4.0
torch-geometric 1.4.2
torch-spline-conv 1.1.1
torch-scatter 2.0.4
torhch-sparse 0.6.0
warmup-scheduler 0.1.1

** Download data **
download the maximal disjoint ball decompositions of ModelNet10 models from
https://drive.google.com/drive/folders/1iVJg94jKGofOjD16yIGvcRprakzduALV?usp=sharing

You should have the following directory structure:

shape_classification
* dataset
   * ModelNet10
* codes
* checkpoints
* results

# run codes
** Train **
go to shape_classification/codes, and run
```
python3 train.py --gcnn_name spline
```
You can use other GraphCNNs, including tag (for TAGConv), and  arma (for ARMAConv).

** Evaluation**
```
python3 eval.py --gcnn_name spline
```
