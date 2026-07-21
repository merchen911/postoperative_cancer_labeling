import os
from datetime import datetime
from typing import Dict


def save_dir_setup(model_name: str, prefix: str = 'CLF-recur', version: str = 'v001', date: str | None = None) -> Dict[str, str]:
    log_dir = os.path.join(
        '..',
        'logs',
        datetime.now().date().isoformat() if date is None else date,
        f'{prefix}-{model_name}-{version}'
    )
    os.makedirs(log_dir, exist_ok=True)

    figdir = os.path.join(log_dir, 'figures')
    os.makedirs(figdir, exist_ok=True)

    filedir = os.path.join(log_dir, 'files')
    os.makedirs(filedir, exist_ok=True)

    return {'figure': figdir, 'file': filedir, 'log': log_dir}
