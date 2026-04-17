# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import nms, ops


class DetectionPredictor(BasePredictor):
    """A class extending the BasePredictor class for prediction based on a detection model.

    This predictor specializes in object detection tasks, processing model outputs into meaningful detection results
    with bounding boxes and class predictions.

    Attributes:
        args (namespace): Configuration arguments for the predictor.
        model (nn.Module): The detection model used for inference.
        batch (list): Batch of images and metadata for processing.

    Methods:
        postprocess: Process raw model predictions into detection results.
        construct_results: Build Results objects from processed predictions.
        construct_result: Create a single Result object from a prediction.
        get_obj_feats: Extract object features from the feature maps.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.detect import DetectionPredictor
        >>> args = dict(model="yolo26n.pt", source=ASSETS)
        >>> predictor = DetectionPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-process predictions and return a list of Results objects.

        This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
        further analysis.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input image tensor in model input format.
            orig_imgs (torch.Tensor | list): Original input images before preprocessing.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): List of Results objects containing the post-processed predictions.

        Examples:
            >>> predictor = DetectionPredictor(overrides=dict(model="yolo26n.pt"))
            >>> results = predictor.predict("path/to/image.jpg")
            >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
        """
        save_feats = getattr(self, "_feats", None) is not None
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=0 if self.args.task == "detect" else len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
            return_idxs=save_feats,
        )

        if getattr(self.args, "pid_nms_enable", False):
            thermal_maps, emissivity_maps = self._get_pid_maps_from_model()
            if save_feats:
                filtered_preds = nms.apply_physical_nms(
                    preds[0],
                    thermal_maps=thermal_maps,
                    emissivity_maps=emissivity_maps,
                    img_tensor=img,
                    conf_thres=self.args.conf,
                    pedestrian_class=getattr(self.args, "pid_pedestrian_class", 0),
                    vehicle_class=getattr(self.args, "pid_vehicle_class", 1),
                    ped_temp_ratio=getattr(self.args, "pid_ped_temp_ratio", 0.90),
                    ped_emissivity_min=getattr(self.args, "pid_ped_emissivity_min", 0.05),
                    vehicle_hot_ratio=getattr(self.args, "pid_vehicle_hot_ratio", 1.05),
                    vehicle_conf_decay=getattr(self.args, "pid_vehicle_conf_decay", 0.9),
                )
                preds = (filtered_preds, preds[1])
            else:
                preds = nms.apply_physical_nms(
                    preds,
                    thermal_maps=thermal_maps,
                    emissivity_maps=emissivity_maps,
                    img_tensor=img,
                    conf_thres=self.args.conf,
                    pedestrian_class=getattr(self.args, "pid_pedestrian_class", 0),
                    vehicle_class=getattr(self.args, "pid_vehicle_class", 1),
                    ped_temp_ratio=getattr(self.args, "pid_ped_temp_ratio", 0.90),
                    ped_emissivity_min=getattr(self.args, "pid_ped_emissivity_min", 0.05),
                    vehicle_hot_ratio=getattr(self.args, "pid_vehicle_hot_ratio", 1.05),
                    vehicle_conf_decay=getattr(self.args, "pid_vehicle_conf_decay", 0.9),
                )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        if save_feats:
            obj_feats = self.get_obj_feats(self._feats, preds[1])
            preds = preds[0]

        results = self.construct_results(preds, img, orig_imgs, **kwargs)

        if save_feats:
            for r, f in zip(results, obj_feats):
                r.feats = f  # add object features to results

        return results

    def _get_pid_maps_from_model(self):
        """Resolve cached PID maps from wrapped model objects."""
        candidates = [self.model, getattr(self.model, "model", None)]
        if getattr(self.model, "model", None) is not None:
            candidates.append(getattr(self.model.model, "model", None))

        for holder in candidates:
            if holder is None:
                continue
            pid_aux = getattr(holder, "_pid_aux", None)
            if isinstance(pid_aux, dict):
                return pid_aux.get("t", None), pid_aux.get("e", None)
        return None, None

    @staticmethod
    def get_obj_feats(feat_maps, idxs):
        """Extract object features from the feature maps."""
        import torch

        s = min(x.shape[1] for x in feat_maps)  # find shortest vector length
        obj_feats = torch.cat(
            [x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, s, x.shape[1] // s).mean(dim=-1) for x in feat_maps], dim=1
        )  # mean reduce all vectors to same length
        return [feats[idx] if idx.shape[0] else [] for feats, idx in zip(obj_feats, idxs)]  # for each img in batch

    def construct_results(self, preds, img, orig_imgs):
        """Construct a list of Results objects from model predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes and scores for each image.
            img (torch.Tensor): Batch of preprocessed images used for inference.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.

        Returns:
            (list[Results]): List of Results objects containing detection information for each image.
        """
        return [
            self.construct_result(pred, img, orig_img, img_path)
            for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0])
        ]

    def construct_result(self, pred, img, orig_img, img_path):
        """Construct a single Results object from one image prediction.

        Args:
            pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
            img (torch.Tensor): Preprocessed image tensor used for inference.
            orig_img (np.ndarray): Original image before preprocessing.
            img_path (str): Path to the original image file.

        Returns:
            (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
        """
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
