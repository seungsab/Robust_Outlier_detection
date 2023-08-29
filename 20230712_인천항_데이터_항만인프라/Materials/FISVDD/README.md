# FISVDD
This package contains the implementation of the Fast Incremental Support Vector Data Descrption (FISVDD) method, an algorithm for online training an SVDD model for outlier detection.

The [paper](https://aaai.org/ojs/index.php/AAAI/article/view/4291) was accepted and presented in the [AAAI-19](https://aaai.org/Conferences/AAAI-19/) conference.

## Reference
If you use this code for your publications, please cite the following paper:

```
@inproceedings{jiang2019fast,
  title={Fast Incremental SVDD Learning Algorithm with the Gaussian Kernel},
  author={Jiang, Hansi and Wang, Haoyu and Hu, Wenhao and Kakde, Deovrat and Chaudhuri, Arin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={3991--3998},
  year={2019}
}
```

## Demo

**Original Data**            | **Training Process**      
:-------------------------:|:-------------------------:
<img src="https://github.com/hs-jiang/FISVDD/blob/master/FISVDD_demo/original_data.png" width="500">  |  <img src="https://github.com/hs-jiang/FISVDD/blob/master/FISVDD_demo/FISVDD_demo.gif" width="500">
**Support Vectors**            |  **Final Result**        
<img src="https://github.com/hs-jiang/FISVDD/blob/master/FISVDD_demo/support_vectors.png" width="500">  |  <img src="https://github.com/hs-jiang/FISVDD/blob/master/FISVDD_demo/final_result.png" width="500">
**Objective Function Value**                              |  
<img src="https://github.com/hs-jiang/FISVDD/blob/master/FISVDD_demo/obv.png" width="500"> |


## License
There is currently a patent pending that covers the FISVDD method. 

For non-commercial or academic use the source code in this package can be distributed and/or modified under the terms of the GNU Lesser General Public License (LGPL) version 3 as published by the Free Software Foundation (http://opensource.org/licenses/lgpl-3.0.html). 

For other usage, please contact the authors. 

