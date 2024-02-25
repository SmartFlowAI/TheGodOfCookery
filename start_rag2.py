import os
os.system('python download_rag2_model.py')
os.system('streamlit run app-enhanced-rag.py --server.address=0.0.0.0 --server.port 7860')
