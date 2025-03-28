datasets = [
    "loris3/babylm_2024_10m_curriculum", # llama done
    # "loris3/stratified_equitoken_10m_curriculum",
    # "loris3/stratified_10m_curriculum"
    # "loris3/babylm_2025_10m_curriculum", TODO
]
curricula = [
   "random.pt",
   "source_difficulty.pt",
   "mattr_increasing.pt",
   "perplexity_increasing.pt"

]

model_types = ["llama"] # "roberta",
__all__ = ["datasets", "curricula","model_types"]