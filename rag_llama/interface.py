import sys
sys.path.append('..')
import torch
from load_rag_engine import load_query_engine


query_engine = load_query_engine()


@torch.inference_mode()
def generate_interactive_rag(
    q: str
):
    return query_engine.query(q)

