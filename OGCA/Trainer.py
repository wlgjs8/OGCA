import logging
import os
from collections import OrderedDict
import pickle

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import hooks
from detectron2.engine.train_loop import SimpleTrainer
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from fvcore.nn.precise_bn import get_bn_modules
from torch.nn.parallel import DistributedDataParallel

from detectron2.data import transforms as T
from detectron2.data import DatasetMapper

from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
)
from detectron2.data import MetadataCatalog
from detectron2.utils.events import get_event_storage

import util.metrics as M
import copy
from tqdm import tqdm
import cv2
import torch.nn.functional as nn
import numpy as np

class SaliencyTrainer(SimpleTrainer):
    """
    A trainer with default training logic. Compared to `SimpleTrainer`, it
    contains the following logic in addition:
    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists.
    3. Register a few common hooks.
    It is created to simplify the **standard model training workflow** and reduce code boilerplate
    for users who only need the standard training workflow, with standard features.
    It means this class makes *many assumptions* about your training logic that
    may easily become invalid in a new research. In fact, any assumptions beyond those made in the
    :class:`SimpleTrainer` are too much for research.
    The code of this class has been annotated about restrictive assumptions it mades.
    When they do not work for you, you're encouraged to:
    1. Overwrite methods of this class, OR:
    2. Use :class:`SimpleTrainer`, which only does minimal SGD training and
       nothing else. You can then add your own hooks if needed. OR:
    3. Write your own training loop similar to `tools/plain_train_net.py`.
    Also note that the behavior of this class, like other functions/classes in
    this file, is not stable, since it is meant to represent the "common default behavior".
    It is only guaranteed to work well with the standard models and training workflow in detectron2.
    To obtain more stable behavior, write your own training logic with other public APIs.
    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    Examples:
    .. code-block:: python
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """

        self.image_size = 384
        self.resize_size = self.image_size + 64

        self.saliency_dataset_mapper = DatasetMapper(cfg, True, augmentations=[
            T.Resize((self.resize_size, self.resize_size)),
            T.RandomCrop('absolute', (self.image_size, self.image_size)),
            T.RandomFlip(),
            T.Resize((self.image_size, self.image_size))
        ])
        self.test_mapper = DatasetMapper(cfg, False, augmentations=[
            T.Resize((self.image_size, self.image_size))
        ])

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg, self.saliency_dataset_mapper)

        # For training, wrap with DDP. But don't need this for inference.
        # if comm.get_world_size() > 1:
        #     model = DistributedDataParallel(
        #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        #     )
        super().__init__(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        # Assume no other objects need to be checkpointed.
        # We can later make it checkpoint the stateful hooks
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True`, and last checkpoint exists, resume from it.
        Otherwise, load a model specified by the config.
        Args:
            resume (bool): whether to do resume or not
        """
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration (or iter zero if there's no checkpoint).
        self.start_iter = (
            self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                "iteration", -1
            )
            + 1
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder='inference'):
        return build_evaluator(cfg, dataset_name, output_folder)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """

        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_test_loader(cfg, self.test_mapper),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model, self.test_mapper, self.iter)

            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers()))
            # ret.append(hooks.PeriodicWriter(self.build_writers()), period=20)
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        It is now implemented by:
        .. code-block:: python
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
        """
        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        
        '''
        import pdb
        try:
            super().train(self.start_iter, self.max_iter)
        except Exception as err:
            pdb.set_trace()
        '''

        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_train_loader(cls, cfg, mapper):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """

        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, test_mapper):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, test_mapper)

    @classmethod
    def test(cls, cfg, model, test_mapper, current_iter, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name, test_mapper)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
            
        return results

def build_evaluator(self, dataset_name, output_dir=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """

    if output_dir is None:
        output_dir = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_list.append(SOCEvaluator(dataset_name, output_dir=output_dir))

    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

class SOCEvaluator(DatasetEvaluator):
    def __init__(self, 
        dataset_name='coco_val_SOC',
        tasks=None,
        distributed=True,
        output_dir=None,
        *,
        max_dets_per_image=None,
        use_fast_impl=True,
        kpt_oks_sigmas=(),
        allow_cached_coco=True,):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._use_fast_impl = False
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        # self._cpu_device = torch.device("cuda")

        # self._predictions

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances
            
            if len(prediction) > 1:
                self._predictions.append(prediction)
            # with open('{}.pickle'.format(prediction['image_id'][:-4]), 'wb') as fw:
            #     pickle.dump(prediction, fw)

    def evaluate(self):
        predictions = self._predictions

        # if len(predictions) == 0:
        #     self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
        #     return {}

        self._results = OrderedDict()
        self._eval_mask_predictions(predictions)

        return copy.deepcopy(self._results)

    def _eval_mask_predictions(self, predictions):
        gt_root = os.path.abspath('/workspace/GLCA_SOC/datasets/SOC/ValSet/gt')
        # gt_root = os.path.abspath('/home/jeeheon/Documents/OGCA/datasets/DUTS/DUTS-TE/DUTS-TE-Mask')
        gt_name_list = sorted(os.listdir(gt_root))

        FM = M.Fmeasure()
        WFM = M.WeightedFmeasure()
        SM = M.Smeasure()
        EM = M.Emeasure()
        MAE = M.MAE()

        for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
            gt_path = os.path.join(gt_root, gt_name)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

            pred_name = gt_name[:-3] + 'jpg'
            preds = self.search_pred(pred_name, predictions)

            mask_prediction = np.zeros(shape=gt.shape, dtype=np.uint8)
            for pred in preds:
                mask_prediction = mask_prediction | pred

            if mask_prediction.shape != gt.shape:
                mask_prediction = nn.interpolate(mask_prediction, size=gt.shape, mode='bilinear', align_corners=False)

            mask_prediction *= 255

            FM.step(pred=mask_prediction, gt=gt)
            WFM.step(pred=mask_prediction, gt=gt)
            SM.step(pred=mask_prediction, gt=gt)
            EM.step(pred=mask_prediction, gt=gt)
            MAE.step(pred=mask_prediction, gt=gt)

        fm = FM.get_results()['fm']
        wfm = WFM.get_results()['wfm']
        sm = SM.get_results()['sm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']

        self._results['SoC'] = {
            'MAE' : mae.round(3),
            'meanEm' : '-' if em['curve'] is None else em['curve'].mean().round(3),
            'maxEm' : '-' if em['curve'] is None else em['curve'].max().round(3),
            'Smeasure' : sm.round(3),
            'wFmeasure' : wfm.round(3)
        }

        storage = get_event_storage()
        storage.put_scalar('MAE', self._results['SoC']['MAE'])
        storage.put_scalar('meanEm', self._results['SoC']['meanEm'])
        storage.put_scalar('maxEm', self._results['SoC']['maxEm'])
        storage.put_scalar('Smeasure', self._results['SoC']['Smeasure'])
        storage.put_scalar('wFmeasure', self._results['SoC']['wFmeasure'])

    def search_pred(self, search_image_name, predictions=None):
        for each_dict in predictions:
            if each_dict['image_id'] == search_image_name:
                return each_dict['instances'].pred_masks.numpy()

        # with open(search_image_name[:-3] + 'pickle', 'rb') as fr:
        #     each_dict = pickle.load(fr)
        # return each_dict['instances'].pred_masks.numpy()
        return None