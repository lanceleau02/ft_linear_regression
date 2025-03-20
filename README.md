# ft_linear_regression

## Virtual environment creation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Virtual environment deletion

```bash
source venv/bin/activate
pip freeze | xargs pip uninstall -y
deactivate
rm -rf venv
```
