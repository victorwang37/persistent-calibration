"""Model-specific checkpoint groups used by calibration evaluators."""

OLMO_MODEL = 'allenai/Olmo-3-1025-7B'
OLMO_32B_MODEL = 'allenai/Olmo-3-1125-32B'
MARIN_MODEL = 'marin-community/marin-8b-base'

MODEL_EVALUATION_REVISIONS = {
    OLMO_MODEL: (
        'stage1-step566000', 'stage1-step707000',
        'stage1-step1272000', 'stage1-step1413814',
    ),
    OLMO_32B_MODEL: (
        'stage1-step262000', 'stage1-step328000',
        'stage1-step590120', 'stage1-step656000',
    ),
    MARIN_MODEL: ('phoenix', 'starling', 'deeper-starling'),
}

# Checkpoints available to the non-oracle methods. Single-checkpoint methods
# use the final checkpoint in each sequence.
MODEL_NON_ORACLE_REVISIONS = {
    OLMO_MODEL: ('stage1-step141000', 'stage1-step283000', 'stage1-step424000'),
    OLMO_32B_MODEL: ('stage1-step66000', 'stage1-step131000', 'stage1-step197000'),
    MARIN_MODEL: ('kestrel', 'ocelot', 'jellyfish'),
}


def evaluation_revisions_for_model(model_name):
    """Return the ordered evaluation checkpoints for *model_name*."""
    try:
        return MODEL_EVALUATION_REVISIONS[model_name]
    except KeyError as exc:
        supported = ', '.join(sorted(MODEL_EVALUATION_REVISIONS))
        raise ValueError(
            f'No evaluation revisions configured for {model_name!r}. '
            f'Supported models: {supported}') from exc


def non_oracle_revisions_for_model(model_name):
    """Return the ordered non-oracle checkpoints for *model_name*."""
    try:
        return MODEL_NON_ORACLE_REVISIONS[model_name]
    except KeyError as exc:
        supported = ', '.join(sorted(MODEL_NON_ORACLE_REVISIONS))
        raise ValueError(
            f'No non-oracle revisions configured for {model_name!r}. '
            f'Supported models: {supported}') from exc
