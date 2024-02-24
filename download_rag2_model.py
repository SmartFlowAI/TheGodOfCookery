import os
# download RAG model
print("Download bce embudding base model")
if not os.path.exists(os.environ.get('HOME') + '/models/bce-embedding-base_v1'):
    command_str = 'huggingface-cli download --token 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  --resume-download maidalun1020/bce-embedding-base_v1 --local-dir-use-symlinks False --local-dir '+ os.environ.get('HOME') + '/models/bce-embedding-base_v1'
    os.system(command_str)

print("Download bce reranker base model")
if not os.path.exists(os.environ.get('HOME') + '/models/bce-reranker-base_v1'):
    command_str = 'huggingface-cli download --token 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  --resume-download maidalun1020/bce-reranker-base_v1 --local-dir-use-symlinks False --local-dir '+ os.environ.get('HOME') + '/models/bce-reranker-base_v1'
    os.system(command_str)
