streamlit run UPDATEDssa165.py


pip install streamlit plotly skyfield numpy pandas requests


wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb

32비트 wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm.deb


sudo dpkg -i cloudflared-linux-*.deb


cloudflared tunnel --url http://localhost:8501
