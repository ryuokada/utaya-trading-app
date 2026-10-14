import sys

if __name__ == "__main__":
  sys.argv = ["streamlit", "run", "utaya.py"]
  from streamlit.web.cli import main

  main()
