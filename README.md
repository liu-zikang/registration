## Instructions
This code has been tested on 
- Python 3.6.10, PyTorch 1.2.0, CUDA 10.2, GeForce RTX 2080Ti/GeForce GTX 1080ti.

### Requirements
To create a virtual environment and install the required dependences please run:
```shell
conda create -n registration
conda activate registration
pip install -r requirements.txt
```
in your working folder.

**Note**: If you want to get the same results as in the paper, install numpy.__version__=='1.19.2' and scipy.__version__=='1.5.0'.

### Datasets and pretrained models

For ModelNet40, the data will be downloaded automatically.

#### Train
```shell
sh 1_experiment_train.sh
```

#### Eval
```shell
sh 1_experiment_eval.sh
```

### Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@article{registration,
  title={local feature guidance framework for robust 3D registration},
  author={Zikang Liu, Kai He, Dazhuang Zhang, Lei Wang},
  journal={The Visual Computer},
  year={2022}
}
```

### Acknowledgments
In this project we use (parts of) the official implementations of the followin works: 

- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [RPMNet](https://github.com/yewzijian/RPMNet) 
- [PCA-GM](https://github.com/Thinklab-SJTU/PCA-GM.git)
- [RGM](https://github.com/fukexue/RGM)

 We thank the respective authors for open sourcing their methods.
