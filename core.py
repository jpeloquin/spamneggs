class Parameter:
    """A random parameter with optional reference levels."""

    def __init__(self, distribution, levels: dict = None):
        self.distribution = distribution
        if levels is None:
            levels = {}
        self.levels = levels
