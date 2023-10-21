# DeepTrack: Lightweight Deep Learning for Vehicle Trajectory Prediction in Highways
Welcome to the DeepTrack GitHub repository, a cutting-edge solution for vehicle trajectory prediction in the realm of intelligent transportation systems (ITS).  DeepTrack is a novel deep learning algorithm specifically tailored for real-time vehicle trajectory prediction. It uses Temporal Convolutional Networks (TCNs)  to encode vehicle dynamics, ensuring high fidelity in time prediction and surpassing traditional methods. It also utilizes Depthwise Convolution that reduces model size and computational complexity, making DeepTrack lightweight yet powerful.

Despite its compact design, DeepTrack delivers accuracy levels that rival leading trajectory prediction models.  With its reduced computational demands, DeepTrack is ideal for deployment on IoT devices in dynamic traffic scenarios. 

# DeepTrack: Vehicle Trajectory Prediction

Welcome to the DeepTrack GitHub repository, a state-of-the-art solution for vehicle trajectory prediction leveraging deep learning techniques.

![Figure 1](./images/figure1.png)

**Dataset for Evaluation**: DeepTrack is evaluated using the NGSIM dataset, as showcased in the CS-LSTM method by N. Deo.

## Training DeepTrack
To train the DeepTrack model, you'll need the `train.mat`, `val.mat`, and `test.mat` files. These files should be generated following the method illustrated by CS-LSTM.

**Training Command**:
```bash
python3.8 train.py
```

## Testing
To evaluate the trained model, use the test.mat file with the following command:
**Testing Command**:
```bash
python3.8 evaluate.py
```


If you use DeepTrack in your research, please cite:
```
@ARTICLE{9770480,
  author={Katariya, Vinit and Baharani, Mohammadreza and Morris, Nichole and Shoghli, Omidreza and Tabkhi, Hamed},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  title={DeepTrack: Lightweight Deep Learning for Vehicle Trajectory Prediction in Highways},
  year={2022},
  volume={23},
  number={10},
  pages={18927-18936},
  doi={10.1109/TITS.2022.3172015}
}
```
