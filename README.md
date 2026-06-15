# GasUNet: Edge-Guided Dual-Branch Mamba with Cross-Scale Fusion for Gas Leak Segmentation
This is the official code of GasUNet: Edge-Guided Dual-Branch Mamba with Cross-Scale Fusion for Gas Leak Segmentation
## InfraGasSegDataset
We propose a high-quality real-world IGS dataset, containing 6426 images and 7390 segmentation targets.
### Dataset Access  
Due to the dataset being collected from real industrial scenarios, the **training dataset is available upon request**. Please send an email to **fanbuqing@stu.hfuu.edu.cn**, clearly stating your purpose of use. 
Thank you for your understanding and support of our work!  
the **InfraGasSegDataset** folder like:
- `InfraGasSegDataset/`: Root directory for the dataset.
  - `img_dir/`: Contains the input infrared gas images.
    - `train/`: Training set images.
    - `val/`: Validation set images.
  - `ann_dir/`: Contains the corresponding annotated masks.
    - `train/`: Masks for training set.
    - `val/`: Masks for validation set.



For detailed setup instructions, we recommend referring to the [MMSegmentation repository](https://github.com/open-mmlab/mmsegmentation). Our test environment is torch2.3.1+cu11.8 and mmcv2.2.0.
```
conda create -n GasUNet python==3.10
conda activate GasUNet
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install mmengine==0.10.5
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.3/index.html
cd GasUNet
pip install -v -e .
```
### Train
```
python tools/train.py configs/GasUNet/GasUNet.py
```
### Test
```
python tools/test.py configs/GasUNet/GasUNet.py models/GasUNet.pth
```
## Contact   
For any question, feel free to email <fanbuqing@stu.hfuu.edu.cn>

### Acknowledgments
We would like to thank the developers of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for their open-source contributions, which greatly supported the development of our work.
Please give us a STAR if the dataset and code help you!

