# Restore the adversarial workflow

Copy or extract this archive into the `AdvTabDDPM_V2` project root.

It restores:

- `attack.py`
- `classify.py`
- `defend_diff_attacks.py`
- clean NPY inputs used by `attack.py`
- pre-generated FGSM, PGD, and C&W adversarial examples and reports
- the small `combine` evaluation set
- `requirements-adversarial.txt`

Install the additional dependency:

```bat
pip install -r requirements-adversarial.txt
```

The current `attack.py` chooses the attack with:

```python
attack_method = "fgsm"  # or "pgd" / "cw"
```

Then run:

```bat
python attack.py
```

Generated files are written to:

```text
data\colosseum\adversarial_examples
```

The restored pre-generated subfolders are:

```text
FGSM
PGD
CW
```
