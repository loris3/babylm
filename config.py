datasets = [
   "loris3/babylm_2024_10m_curriculum", # llama done, roberta done
   "loris3/stratified_equitoken_10m_curriculum",
   "loris3/stratified_10m_curriculum"
]
baseline_curricula = [
   "random.pt",
   "source_difficulty.pt",
   "mattr_increasing.pt",
   "perplexity_increasing.pt"

]
influence_curricula = [
    "_incr_influence_epoch_repetition.pt",
   "_influence_epoch_repetition.pt",
   "_influence_incr_bins_dirac.pt",
   "_influence_decr_bins_dirac.pt",

   "_influence_incr_bins_lognorm.pt",
   "_influence_decr_bins_lognorm.pt",

   "_influence_incr_cp_dirac.pt",
   "_influence_decr_cp_dirac.pt",
   "_influence_top_50_cp_shuffled.pt",
   "_influence_tracin_sandwich.pt",
]


model_types = ["llama","roberta"]

baseline_models = [
   ("BabyLM-community/babylm-baseline-10m-gpt-bert-mixed", "roberta"), # model_type just for the eval scripts to set eval mode mlm
   ("BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus", "roberta"),
   ("BabyLM-community/babylm-baseline-10m-gpt-bert-masked-focus", "roberta"),
   ("BabyLM-community/babylm-baseline-10m-gpt2", "llama")
]
__all__ = ["datasets", "curricula","model_types"]