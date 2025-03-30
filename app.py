import os
import subprocess

# Vérifie si Streamlit est installé, sinon l'installe
try:
    import streamlit
except ImportError:
    print("Streamlit n'est pas installé. Installation en cours...")
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "streamlit"])

# Lancer l'application Streamlit
subprocess.run(["streamlit", "run", "script.py"])