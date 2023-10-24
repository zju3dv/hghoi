import yacs.config


class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config


_C = CN()

_C.TYPE = "DefaultTrainer"

STATE_DIM = 647
ACTION_NUMBER = 5
PART_N = 5
QUANT_N = 8
GOAL_DIM = 3

CODE_DIM = PART_N * QUANT_N
CODE_DIM = 20
OFFSET_DIM = 66
INTEGRAL_FRAMES = 5

JOINTS_NUM = 22
LOCAL_POSE_DIM = JOINTS_NUM * 12
INVERSE_POSE_DIM = JOINTS_NUM * 3 + LOCAL_POSE_DIM
# pelvis, right_wrist, left_wrist, right_ankle and left_ankle
CONTACT_DIM = 5 + INVERSE_POSE_DIM


DATASET = CN()
DATASET.TYPE = "PoseSequencePositionData"
dataset_cfg = CN()
dataset_cfg.debug = False
dataset_cfg.data_dir = "./datasets/samp/PIDCodeMotionNet"
dataset_cfg.L = 60

dataset_cfg.max_len = 5000
dataset_cfg.division = 4

dataset_cfg.is_pred_absolute_position = False
dataset_cfg.is_pred_invtraj = False

dataset_cfg.is_goalpose = False
dataset_cfg.is_env = False
dataset_cfg.is_minmax = False
dataset_cfg.pose_dim = 264
dataset_cfg.start_traj = 330
dataset_cfg.end_traj = 447
dataset_cfg.contact_dim = 5
dataset_cfg.interaction_dim = 2048
dataset_cfg.env_dim = 315

dataset_cfg.predict_extraframes = 0
dataset_cfg.pred_time_range = 900

dataset_cfg.collate_fn = "default_collate"
dataset_cfg.forward_batch_size = 1
dataset_cfg.forward_collate_fn = "remove_batch_collate"
dataset_cfg.valid_collate_fn = "default_collate"
dataset_cfg.valid_batch_size = 1

DATASET.cfg = dataset_cfg


DATALOADER = CN()
DATALOADER.batch_size = 32
DATALOADER.num_workers = 2

MODEL = CN()
MODEL.TYPE = "DDPMMotion"

MODEL.cfg = CN()
MODEL.cfg.ENCODER_TYPE = "VQContactEnvPosePartEncoder"
MODEL.cfg.DECODER_TYPE = "VQContactEnvPosePartDecoder"

MODEL.cfg.QUANTIZE_TYPE = "QuantizeLayer"

MODEL.cfg.state_dim = STATE_DIM
MODEL.cfg.interaction_dim = 2048
MODEL.cfg.z_dim = 64
MODEL.cfg.act = "Exp"
MODEL.cfg.dropout = 0.0

MODEL.cfg.pose_dim = 198
MODEL.cfg.joint_dim = 9
MODEL.cfg.action_dim = ACTION_NUMBER
MODEL.cfg.goal_dim = GOAL_DIM
MODEL.cfg.quant_dim_enc = 8
MODEL.cfg.quant_n = QUANT_N
MODEL.cfg.quant_dim_dec = 40
MODEL.cfg.quant_net = "Quantize"

MODEL.cfg.norm = None
MODEL.cfg.activation = "ReLU"
MODEL.cfg.use_pred_dist_during_test = False

MODEL.cfg.num_heads = 8
MODEL.cfg.part_n = PART_N

MODEL.cfg.input_pose_dim = 198
MODEL.cfg.latent_dim = 128
MODEL.cfg.ff_size = 128
MODEL.cfg.num_layers = 4
MODEL.cfg.embedding_shape = -1
MODEL.cfg.mask_token = False
MODEL.cfg.positional_encoding_type = "PositionalEncoding"

MODEL.cfg.pred_time_range = 900
MODEL.cfg.root_dim = 4
MODEL.cfg.traj_dim = 4

MODEL.cfg.env_dim = 315

MODEL.cfg.quant_key_list = []
MODEL.cfg.pred_n = 4

MODEL.cfg.contact_dim = 5
MODEL.cfg.input_state_dim = 122

MODEL.cfg.self_condition = True
MODEL.cfg.objective = "pred_noise"  # {'pred_noise' or 'pred_x0'}
MODEL.cfg.num_timesteps = 1000
MODEL.cfg.sampling_timesteps = 1000
MODEL.cfg.beta_schedule = "cosine"
MODEL.cfg.p2_loss_weight_gamma = 0.0
MODEL.cfg.p2_loss_weight_k = 1.0
MODEL.cfg.ddim_sampling_eta = 1.0
MODEL.cfg.is_minmax = False
MODEL.cfg.self_condition_prob = 0.5

MODEL.cfg.L = 60

MODEL.cfg.mask_cond_prob = 0.0
MODEL.cfg.guidance_scale = 2.5

PIPELINE = CN()
PIPELINE.TYPE = "DDPMPipeline"
PIPELINE.cfg = CN()
PIPELINE.cfg.used_loss = "l2"

PIPELINE.cfg.func_wrap_pairs = []


LOSS = CN()
RECONSTRUCTION = CN()
RECONSTRUCTION.TYPE = "mseloss"
RECONSTRUCTION.weight = 1.0
RECONSTRUCTION.reduction = "sum"
RECONSTRUCTION.average_dim = [0]
LOSS.RECONSTRUCTION = RECONSTRUCTION

KLD = CN()
KLD.TYPE = "kldloss"
KLD.weight = 0.1
KLD.reduction = "sum"
KLD.average_dim = [0]
LOSS.KLD = KLD

LATENT = CN()
LATENT.TYPE = "l1loss"
LATENT.weight = 0.25
LATENT.reduction = "mean"
LOSS.LATENT = LATENT

NLL = CN()
NLL.TYPE = "crossentropyloss"
NLL.weight = 1.0
LOSS.NLL = NLL

L2 = CN()
L2.TYPE = "weightedmseloss"
L2.weight = 1.0
LOSS.L2 = L2

L1 = CN()
L1.TYPE = "weightedl1loss"
L1.weight = 1.0
LOSS.L1 = L1

PIPELINE.cfg.LOSS = LOSS


CRITERION = CN()
CRITERION.TYPE = "PoseNetCriterion"
CRITERION.weight = 1.0
CRITERION.save_results = True
CRITERION.external_model_config = None
CRITERION.cfg = CN()
CRITERION.cfg.quant_dim = 4
CRITERION.cfg.is_pred_absolute_position = False
CRITERION.cfg.is_pred_invtraj = False
CRITERION.cfg.is_Milestone = False
CRITERION.cfg.is_minmax = False


OPTIMIZER = CN()
OPTIMIZER.TYPE = "Adam"
OPTIMIZER.lr = 5e-5


SCHEDULE = CN()
SCHEDULE.max_epochs = 100
SCHEDULE.lr = 1e-4
SCHEDULE.min_lr = 5e-6
scheduler = CN()
scheduler.TYPE = "LambdaLR"
scheduler.lr_lambda = "linear_epoch_func"
# SCHEDULE.scheduler = scheduler


HOOK = CN()
hook_save = CN()
hook_save.TYPE = "SaveCkptHook"
hook_save.priority = 10
hook_save.interval = 100
HOOK.hook_save = hook_save
hook_ss = CN()
hook_ss.TYPE = "ScheduledSamplingHook"
hook_ss.priority = 20
hook_ss.milestones = [30, 60]
# HOOK.hook_ss = hook_ss
hook_anneal = CN()
hook_anneal.TYPE = "AnnealingHook"
hook_anneal.priority = 30
hook_anneal.annealing_pairs = []
# hook_anneal.annealing_pairs = [["KLD", [0, 25]]]
HOOK.hook_anneal = hook_anneal
hook_clip_grad = CN()
hook_clip_grad.TYPE = "ClipGradHook"
hook_clip_grad.priority = 10
hook_clip_grad.max_grad = None
HOOK.hook_clip_grad = hook_clip_grad
hook_eval = CN()
hook_eval.TYPE = "EvalHook"
hook_eval.priority = 80
hook_eval.interval = 1000
hook_eval.test_before_train = False
HOOK.hook_eval = hook_eval
hook_log = CN()
hook_log.TYPE = "LogShowHook"
hook_log.priority = 90
HOOK.hook_log = hook_log
hook_tb = CN()
hook_tb.TYPE = "TensorBoardHook"
hook_tb.priority = 100
HOOK.hook_tb = hook_tb

FUNC = CN()
FUNC.original_dataset_dir = "./datasets/samp/LocalPose"
FUNC.dataset_save_dir = "./datasets/samp/PoseSequence"
FUNC.save_dir = "./work_dirs/actor/results"

cfg = CN()
cfg.output_dir = "work_dirs/debug"

cfg.DATASET = DATASET
cfg.DATALOADER = DATALOADER
cfg.MODEL = MODEL
cfg.PIPELINE = PIPELINE
cfg.CRITERION = CRITERION
cfg.OPTIMIZER = OPTIMIZER
cfg.SCHEDULE = SCHEDULE
cfg.HOOK = HOOK
cfg.FUNC = FUNC

_C.cfg = cfg
