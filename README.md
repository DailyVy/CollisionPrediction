Test

## Installation
```bash
# recommended to create a new environment with torch1.13 + cuda11.7
# conda environmental setting
conda create -n CollPred python=3.8
conda activate CollPred
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install opencv_python==4.7.0.72

# DepthAnythingV2, Grounding DINO
pip install huggingface_hub
pip install transformer

# unimatch installation
#git clone https://github.com/autonomousvision/unimatch
cd /content/unimatch
# bash pip_install.sh # => 이미 있는 패키지를 제외하고 나머지 설치 (아래 pip)
pip install imageio==2.9.0 imageio-ffmpeg scikit-image scipy tensorboard==2.9.1

# download pretrained models
cd unimatch
mkdir pretrained
wget -P pretrained https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
wget -P pretrained https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth
wget -P pretrained https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmstereo-scale2-regrefine3-resumeflowthings-middleburyfthighres-a82bec03.pth
wget -P pretrained https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth
```


### CAT-Seg + SAM + YOLO + OF
1. CAT-Seg를 통해 floor를 인식 (CAT-Seg github에서 weight download 필요)
2. Segment Anything의 prompt를 grid point로 하되, floor에 해당하는 부분만 masking하여 multi mask를 얻어냄
3. YOLO로 검출된 Hoist hook 아래에 있는 mask만 filtering - candiadate masks
4. Optical flow를 측정하여 특정 threshold(default=1.0) 이하는 제외
```bash
python distance.py --input /path/to/image --output /path/to/save/ --opts MODEL.WEIGHTS CAT-Seg/model_large.pth
```

### CAT-Seg + FastSAM + YOLO + OF
1. CAT-Seg를 통해 floor를 인식 (CAT-Seg github에서 weight download 필요)
2. FastSAM로 multi masks를 구하고, floor에 해당하는 부분만 filtering하여 filtered masks를 얻어냄
3. YOLO로 검출된 Hoist hook 아래에 있는 mask만 filtering - candiadate masks
4. Optical flow를 측정하여 특정 threshold(default=1.0) 이하는 제외
```bash
python distance_fastsam.py --input /path/to/image --output /path/to/save/ --opts MODEL.WEIGHTS CAT-Seg/model_large.pth
```
