import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.system('huggingface-cli download --resume-download moka-ai/m3e-base '
          '--local-dir /home/xlab-app-center/models/m3e-base')
os.system('streamlit run app.py --server.address=0.0.0.0 --server.port 7860')