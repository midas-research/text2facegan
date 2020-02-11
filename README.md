# Text2FaceGAN: Face Generation from Fine Grained Textual Descriptions

This repository contains code and data used in the paper "Text2FaceGAN: Face Generation from Fine Grained Textual Descriptions". This work was done by students and collaborators at:

<br>
<p align="center">
  <img src="https://github.com/midas-research/bhaav/blob/master/MIDAS-logo.jpg" alt="MIDAS lab at IIIT-Delhi"  width="60%"/>
  <br>
</p>
<br>

## Dataset

We leverage the celebA dataset to create a dataset that contains face images with text captions. In `data/` directory you will find `caps.txt` which contains a list of captions. The first column of each row consists of the image name and the second column is the captions. In the second column each caption is seperated by a `|` (please refer to the paper to find what each caption signifies and how these captions were generated). 

Please refer to the paper for more information.


## References

Please cite [[1]](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8919389&isnumber=8919254) if you found the resources in this repository useful. Preprint of the paper can be found here: [https://arxiv.org/abs/1911.11378](https://arxiv.org/abs/1911.11378)

[1] O. R. Nasir, S. K. Jha, M. S. Grover, Y. Yu, A. Kumar and R. R. Shah, "[Text2FaceGAN: Face Generation from Fine Grained Textual Descriptions,](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8919389&isnumber=8919254)" 2019 IEEE Fifth International Conference on Multimedia Big Data (BigMM), Singapore, Singapore, 2019, pp. 58-67.
doi: 10.1109/BigMM.2019.00-42.

```
@inproceedings{nasir2019text2facegan,
  author={O. R. {Nasir} and S. K. {Jha} and M. S. {Grover} and Y. {Yu} and A. {Kumar} and R. R. {Shah}},
  booktitle={2019 IEEE Fifth International Conference on Multimedia Big Data (BigMM)},
  title={Text2FaceGAN: Face Generation from Fine Grained Textual Descriptions},
  year={2019},
  pages={58-67},
  doi={10.1109/BigMM.2019.00-42},
  month={Sep.},
}
```
