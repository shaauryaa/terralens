"""TerraLens: multilingual stance detection on Hindi and Bengali YouTube comments.

Target claim: "Climate change and global warming is a serious concern."
Labels: Favour / Against / None.
"""

__version__ = "0.1.0"

LABELS = ["Favour", "Against", "None"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
