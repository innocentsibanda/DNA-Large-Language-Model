# Central Configurations 

# Data
DATA_FILES = ["Data/complete_sequences.txt"]  
PARTIAL_DATA_FILES = ["Data/partial_sequences.txt"]  
STAGE2_RIBOZYME_FULL_FILES = ["Data/ribo_full.txt"]
STAGE3_RIBOZYME_PARTIAL_FILES = ["Data/ribo_partial.txt"]
STAGE4_WEAK_SEQCLS = ["Data/weak_seq_class_partial.txt"]
STAGE5_WEAK_RIBOZYME_SEQCLS = ["Data/weak_ribozyme_seqcls.txt"]
STAGE6_GOLD_MIXED = ["Data/gold_motifs_mixed.jsonl"]

# Global sampling for tokenizer & Stage 1a
DATASET_FRACTION = 0.0001
TRAINSET_FRACTION = DATASET_FRACTION
RANDOM_SEED = 42

# Pretraining Tasks 
STAGE1_TASKS = ["mlm", "span", "kmer_reorder", "dae"]
STAGE6_TASKS = ["masked_motif", "mbp", "map", "cml"]

# Pretraining Task Selection
PRETRAIN_TASK = "kmer_reorder"   # Select one : "mlm", "span", "kmer_reorder", "dae"

# Tokenizer Selection 
TOKENIZER_TYPE = "bpe"   # "kmer", "bpe", or "hybrid"

# Hybrid mode
HYBRID_MODE = "token"       # "dual" = two streams; "token" = interleaved single stream

# For dual-stream self-supervision select which stream receives labels?
SELF_SUP_STREAM = "kmer"   # "kmer" or "bpe"

# Dual-stream encoder fusion (only when HYBRID_MODE == "dual")
DUAL_FUSION = "sum"        # "sum" or "concat"

# Positional embeddings: "learned" or "sinusoidal"
POSITIONAL_EMBEDDING_TYPE = "sinusoidal"

# TXT format supported with txt_schema = seq_label
MAP_NUM_CLASSES_COARSE = 8           # number of coarse classes for example (gag, pol, env)
MAP_LABEL_KEY = "global_label"       # the field carrying class id
MAP_NUM_CLASSES_RIBOZYME = 5         # for example {hammerhead, hairpin, HDV, VS, glmS}

# Loss mixing for Stage 4 & 5
CE_WEIGHT = 1.0
SUPCON_WEIGHT = 0.5
SUPCON_TEMPERATURE = 0.07

# K-mer and bpe parameters
KMER_SIZE = 3
VOCAB_SIZE = 1000
MERGE_NUM = 100

# Special tokens (names in vocab)
PAD_TOKEN  = "[PAD]"
MASK_TOKEN = "[MASK]"
UNK_TOKEN  = "[UNK]"
CLS_TOKEN  = "[CLS]"
SEP_TOKEN  = "[SEP]"

# Fallback PAD id
PAD_TOKEN_ID = 0

# Model & embeddings
EMBED_DIM = 128
NUM_HEADS = 8
FF_DIM = 512
NUM_LAYERS = 4
MAX_LEN = 200
DROPOUT = 0.1
ACTIVATION = "relu"        # Examples "relu" or "gelu"

# Motif channel
MOTIF_VOCAB_SIZE = 2       # for token-level heads 
PROJECTION_DIM = 128       # projection head dimm used by Contrastive head 

# Task-specific knobs for self-supervision
MASK_PROB = 0.15           
SPAN_MAX_LEN = 5           # for "span"
REORDER_WINDOW = 5         # for "kmer_reorder"

# Self-supervised on partial sequences
PARTIAL_ENABLE = True
PARTIAL_WINDOW_SIZE = 150
PARTIAL_WINDOWS_PER_SEQ = 1
PARTIAL_STRATEGY = "random"   
PARTIAL_OVERLAP = 0
EPOCHS_PARTIAL = 5           

# Weak supervision
WEAK_SUP_ENABLE = False
WEAK_SUP_FILES = []
WEAK_LABEL_SOURCE = "heuristic"
WEAK_PSEUDO_PROB_THRESH = 0.9
WEAK_KEEP_UNCERTAIN_AS_IGNORE = False
EPOCHS_WEAK = 5
WEAK_LR_MULT = 0.5

WEAK_HEURISTIC_PATTERNS = {
    "A_rich": r"A{4,}",
    "TATA": r"TATA[AT]A",
}

# Fully supervised 
SUPERVISED_FILES = []
SUPERVISED_TASKS = ["masked_motif", "mbp", ]
SUPERVISED_EPOCHS = 1

# Training Supervised Stages
BATCH_SIZE = 4
EPOCHS = 1       
LEARNING_RATE = 1e-3
GRAD_CLIP = 1.0

# Stages 5–7 Auxiliary self-supervised loss
AUX_TASK = "mlm"             # {"mlm","dae","span","kmer_reorder"}
AUX_LOSS_WEIGHT = 0.1        

# Stage-1  Replay Buffer 
REPLAY_FRACTION = 0.01    

# Encoder layers freezing for Stages 5–7
UNFREEZE_TOP_N = 0           # 0 = fully frozen transformer stack except the embeddings layer

MIXED_PRECISION = True      
USE_EMA = True              
EMA_DECAY = 0.999

WARMUP_STEPS = 100
COSINE_TOTAL_STEPS = 1000    

EARLY_STOP_PATIENCE = 5

# Focal loss
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0
USE_CLASS_WEIGHTS = True 

ADAPTER_ENABLE    = False     
ADAPTER_R         = 16        
ADAPTER_DIM       = ADAPTER_R 
ADAPTER_SCALING   = 1.0       
ADAPTER_NONLINEAR = "relu"    
ADAPTER_DROPOUT   = 0.0

LORA_ENABLE = False
LORA_R      = 8
LORA_ALPHA  = 16
LORA_DROPOUT = 0.0

LORA_TARGETS = [
    "encoder.layers",                
    "self_attn.in_proj_weight",      
    "self_attn.out_proj",           
    "linear1",                       
    "linear2",                       
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "attn_q", "attn_k", "attn_v", "attn_out",
    "ffn_in", "ffn_out",
]


STAGE_GATES = {

    "masked_accuracy_min_delta": 0.02, 
    "ece_max_delta":           -0.01,   
    "reconstruction_min_delta": 0.02,  
    "entropy_stability_eps":    0.01,   
    "require_any_two":          True,  
}

CURRICULUM_ORDER = ["rna_full", "ribo_full", "ribo_partial"]

# Extra epochs for new stages
EPOCHS_STAGE2 = 3     
EPOCHS_STAGE3 = 3     
EPOCHS_STAGE4 = 2
EPOCHS_STAGE5 = 3
EPOCHS_STAGE6 = 3

LR_MULT_STAGE4 = 0.5
LR_MULT_STAGE5 = 0.5
LR_MULT_STAGE6 = 0.2

# Logging 
PROBE_FIRST_BATCH = True
CSV_LOG_PATH = "metrics.csv"

# W&B 
USE_WANDB = False
WANDB_PROJECT = "my-project"
WANDB_ENTITY = None
WANDB_RUN_NAME = None

# Saving tokenizer paths 
SAVE_DIR = "."  
TOKENIZER_SAVE_PATH   = "tokenizer_vocab.json"
TOKENIZER_BPE_PATH    = "bpe_vocab.json"
TOKENIZER_HYBRID_PATH = "hybrid_vocab.json"
TOKENIZER_KMER_PATH   = "kmer_vocab.json"
STAGE_LOG_PATH = "stage_gates.jsonl"          
RUN_CONFIG_SNAPSHOT = "config_snapshot.json"  
TOKENIZER_META_PATH = "tokenizer_meta.json"

# Checkpoints per stage
CKPT_STAGE1A = "stage1a_pretrained.pt"
CKPT_STAGE1B = "stage1b_pretrained.pt"
CKPT_STAGE2  = "stage2_ribo_full_selfsup.pt"
CKPT_STAGE3  = "stage3_ribo_partial_selfsup.pt"
CKPT_STAGE4  = "stage4_seqcls_coarse.pt"
CKPT_STAGE5  = "stage5_seqcls_ribozyme.pt"
CKPT_STAGE6  = "stage6_gold_mixed.pt"
