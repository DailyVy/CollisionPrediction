import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../CAT-Seg"))

import torch

from types import SimpleNamespace as ns

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

from cat_seg import add_cat_seg_config

class CATSegSegmentationMap(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, text=None):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)
        # set classes
        if text is not None:
            pred = self.predictor.model.sem_seg_head.predictor
            pred.test_class_texts = text.split(',')
            pred.text_features_test = pred.class_embeddings(pred.test_class_texts, 
                #imagenet_templates.IMAGENET_TEMPLATES, 
                 ['A photo of a {} in the scene',],
                pred.clip_model).permute(1, 0, 2).float().repeat(1, 80, 1)
            self.metadata = ns()
            self.metadata.stuff_classes = pred.test_class_texts
    
    def run_on_image_custom_text(self, image, text=None):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        
        if text is not None:
            pred = self.predictor.model.sem_seg_head.predictor
            print(text)
            pred.test_class_texts = text.split(',')
            pred.text_features_test = pred.class_embeddings(pred.test_class_texts, 
                #imagenet_templates.IMAGENET_TEMPLATES, 
                 ['A photo of a {} in the scene',],
                pred.clip_model).permute(1, 0, 2).float().repeat(1, 80, 1)
            self.metadata = ns()
            self.metadata.stuff_classes = pred.test_class_texts

        predictions = self.predictor(image)
        segmap = predictions['sem_seg'].argmax(dim=0)
        print(f"type(segmap) : {type(segmap)} \t shape: {segmap.shape}")
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device),
                    alpha=0.4,
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output, segmap

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg