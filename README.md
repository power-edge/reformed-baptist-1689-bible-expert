# Reformed Baptist 1689 Bible Expert 

## Setup virtual environment

Create a virtual environment using specific Python version

```shell
python3.10 -m venv venv
```

Activate the virtual environment

```shell
. ./venv/bin/activate
```

Install dependencies

```shell
pip install --upgrade pip
uv sync --no-install-project --active
```

Run mlserver

```shell
mlserver start .
```

Run test

```shell
python client_example.py
```


Build the image

```shell
mlserver build . -t 'poweredgesports/reformed-baptist-1689-bible-expert-server:0.1.0'
```
