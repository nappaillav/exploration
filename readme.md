# Exploration Repository

This repository contains code that can run Random Network Distillation with Mario and ALE/Montezuma (Atari game).

To run the experiment, it is preferable to have a custom environment set up, such as `virtualenv` or `conda`.

```bash
cd exploration
pip install -r requirements.txt
pip install -e .
```

For Mario, use the following command:

```bash
python mario_ppo.py
```

