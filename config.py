datasets = [
    "loris3/babylm_2024_10m_curriculum",
    "loris3/stratified_equitoken_10m_curriculum",
    "loris3/stratified_10m_curriculum"
    # "loris3/babylm_2025_10m_curriculum", TODO
]
curricula = [
    "random.pt"
]

model_types = ["roberta", "llama"]
__all__ = ["datasets", "curricula",model_types]