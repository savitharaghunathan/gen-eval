class ProfileValidationError(ValueError):
    pass


class UnknownMetricError(ValueError):
    def __init__(self, metric_name: str, available: list[str] | None = None):
        if available:
            msg = f"Unknown metric '{metric_name}'. Available metrics: {', '.join(sorted(available))}"
        else:
            msg = f"Unknown metric '{metric_name}'"
        super().__init__(msg)
        self.metric_name = metric_name
        self.available = available


class ProfileNotFoundError(KeyError):
    def __init__(self, profile_name: str):
        super().__init__(f"Profile not found: '{profile_name}'")
        self.profile_name = profile_name
