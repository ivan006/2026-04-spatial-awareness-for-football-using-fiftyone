install python

run these

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## 📦 Update requirements

```bash
pip freeze | Out-File -Encoding utf8 requirements.txt
```

> (PowerShell only — avoids binary file issue)
