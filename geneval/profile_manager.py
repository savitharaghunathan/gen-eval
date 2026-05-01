import copy
import datetime
import logging
from pathlib import Path

import yaml

from geneval.exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError
from geneval.metric_registry import get_available_metrics, resolve_metric_candidates
from geneval.schemas import BatchResult, MetricEvaluation, ProfileResult

_WEIGHT_TOLERANCE = 1e-6

logger = logging.getLogger(__name__)


class ProfileManager:
    def __init__(self, profiles_path: str | None = None):
        self._profiles: dict[str, dict] = {}
        self._policies: dict[str, dict] = {}

        self._load_builtins()

        if profiles_path is not None:
            path = Path(profiles_path)
            if not path.exists():
                raise FileNotFoundError(f"Profiles file not found: {profiles_path}")
            self._load_user_profiles(path)

        self._validate_all_profiles()

    def _load_builtins(self):
        builtin_path = Path(__file__).parent / "profiles" / "default_profiles.yaml"
        if builtin_path.exists():
            with open(builtin_path) as f:
                data = yaml.safe_load(f)
            if data and "profiles" in data:
                self._profiles.update(data["profiles"])

    def _load_user_profiles(self, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            return
        if "profiles" in data:
            self._profiles.update(data["profiles"])
        if "policies" in data:
            self._policies.update(data["policies"])

    def _validate_all_profiles(self):
        available_metrics = get_available_metrics()
        for name, profile in self._profiles.items():
            self._validate_profile(name, profile, available_metrics)

    def _validate_profile(self, name: str, profile: dict, available_metrics: list[str]):
        metrics = profile.get("metrics", [])
        weights = profile.get("weights", {})
        criteria = profile.get("criteria", {})

        for m in metrics:
            if m not in available_metrics:
                raise UnknownMetricError(m, available=available_metrics)
            if m not in weights:
                raise ProfileValidationError(f"Profile '{name}': metric '{m}' missing from weights")
            if m not in criteria:
                raise ProfileValidationError(f"Profile '{name}': metric '{m}' missing from criteria")

        weight_sum = sum(weights.get(m, 0) for m in metrics)
        if abs(weight_sum - 1.0) > _WEIGHT_TOLERANCE:
            raise ProfileValidationError(f"Profile '{name}': weights sum to {weight_sum}, expected 1.0")

        if "composite_threshold" not in profile:
            default_threshold = sum(weights.get(m, 0) * criteria.get(m, 0) for m in metrics)
            profile["composite_threshold"] = default_threshold

    def get_profile(self, name: str) -> dict:
        if name not in self._profiles:
            raise ProfileNotFoundError(name)
        return copy.deepcopy(self._profiles[name])

    def list_profiles(self) -> list[str]:
        return list(self._profiles.keys())

    def list_policies(self) -> list[str]:
        return list(self._policies.keys())

    def resolve_policy(self, policy_name: str) -> dict:
        if policy_name not in self._policies:
            raise ProfileNotFoundError(policy_name)

        policy = self._policies[policy_name]
        base_profile_name = policy["profile"]

        if base_profile_name not in self._profiles:
            raise ProfileNotFoundError(base_profile_name)

        profile = copy.deepcopy(self._profiles[base_profile_name])

        overrides = policy.get("overrides", {})
        if "criteria" in overrides:
            profile["criteria"].update(overrides["criteria"])
        if "composite_threshold" in overrides:
            profile["composite_threshold"] = overrides["composite_threshold"]

        return profile

    def _compute_profile_result(
        self,
        profile_name: str,
        policy_name: str | None,
        profile: dict,
        scores: dict[str, tuple],
    ) -> ProfileResult:
        metrics = profile["metrics"]
        weights = profile["weights"]
        criteria = profile["criteria"]
        composite_threshold = profile["composite_threshold"]

        metric_results = []
        for m in metrics:
            raw_score, adapter, details = scores[m]
            score = raw_score if raw_score is not None else 0.0
            threshold = criteria[m]
            weight = weights[m]
            passed = score >= threshold if raw_score is not None else False

            metric_results.append(
                MetricEvaluation(
                    name=m,
                    score=score,
                    threshold=threshold,
                    passed=passed,
                    weight=weight,
                    weighted_score=weight * score,
                    adapter=adapter,
                    details=details,
                )
            )

        composite_score = sum(mr.weighted_score for mr in metric_results)
        composite_passed = composite_score >= composite_threshold
        overall_passed = composite_passed and all(mr.passed for mr in metric_results)

        return ProfileResult(
            profile_name=profile_name,
            policy_name=policy_name,
            overall_passed=overall_passed,
            composite_score=composite_score,
            composite_threshold=composite_threshold,
            composite_passed=composite_passed,
            metric_results=metric_results,
            metadata={
                "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
            },
        )

    def _run_metrics(self, framework, profile: dict, question: str, response: str, reference: str, retrieval_context: str) -> dict[str, tuple]:
        scores = {}
        for metric_name in profile["metrics"]:
            candidates = resolve_metric_candidates(metric_name)
            # Filter to adapters that actually support this metric at runtime
            adapters = getattr(framework, "adapters", None)
            if isinstance(adapters, dict):
                candidates = [(name, cls) for name, cls in candidates if name in adapters and metric_name in adapters[name].supported_metrics]
            resolved = False
            for adapter_name, adapter_metric_class in candidates:
                try:
                    raw_results = framework.evaluate(
                        question=question,
                        response=response,
                        reference=reference,
                        retrieval_context=retrieval_context,
                        metrics=[f"{adapter_name}.{adapter_metric_class}"],
                    )
                    key = f"{adapter_name}.{adapter_metric_class}"
                    if key in raw_results:
                        _, output = raw_results[key]
                        if output.metrics:
                            result_metric = output.metrics[0]
                            scores[metric_name] = (result_metric.score, adapter_name, result_metric.details)
                            resolved = True
                            break
                except Exception as e:
                    logger.warning(f"Adapter '{adapter_name}' failed for metric '{metric_name}': {e}")
            if not resolved:
                last_adapter = candidates[-1][0] if candidates else "unknown"
                scores[metric_name] = (None, last_adapter, None)
        return scores

    def evaluate(
        self,
        framework,
        profile_name: str | None = None,
        policy_name: str | None = None,
        question: str = "",
        response: str = "",
        reference: str = "",
        retrieval_context: str = "",
    ) -> ProfileResult:
        if policy_name:
            profile = self.resolve_policy(policy_name)
            effective_profile_name = self._policies[policy_name]["profile"]
        elif profile_name:
            profile = copy.deepcopy(self.get_profile(profile_name))
            effective_profile_name = profile_name
        else:
            raise ValueError("Either profile_name or policy_name must be provided")

        scores = self._run_metrics(framework, profile, question, response, reference, retrieval_context)
        return self._compute_profile_result(effective_profile_name, policy_name, profile, scores)

    def evaluate_batch(
        self,
        framework,
        data_path: str,
        profile_name: str | None = None,
        policy_name: str | None = None,
    ) -> BatchResult:
        with open(data_path) as f:
            data = yaml.safe_load(f)

        test_cases = data.get("test_cases", [])
        if not test_cases:
            raise ValueError(f"No test cases found in {data_path}")

        case_results = []
        for case in test_cases:
            result = self.evaluate(
                framework=framework,
                profile_name=profile_name,
                policy_name=policy_name,
                question=case.get("user_input", ""),
                response=case.get("response", ""),
                reference=case.get("reference", ""),
                retrieval_context=case.get("retrieved_contexts", ""),
            )
            case_results.append(result)

        all_passed = all(cr.overall_passed for cr in case_results)
        pass_rate = sum(1 for cr in case_results if cr.overall_passed) / len(case_results)

        summary = {}
        if case_results:
            all_metric_names = {mr.name for cr in case_results for mr in cr.metric_results}
            for mn in all_metric_names:
                metric_scores = [mr.score for cr in case_results for mr in cr.metric_results if mr.name == mn]
                summary[mn] = {"mean": sum(metric_scores) / len(metric_scores) if metric_scores else 0.0}

        if policy_name:
            effective_profile_name = self._policies[policy_name]["profile"]
        else:
            effective_profile_name = profile_name or ""

        return BatchResult(
            profile_name=effective_profile_name,
            policy_name=policy_name,
            overall_passed=all_passed,
            case_results=case_results,
            summary=summary,
            pass_rate=pass_rate,
            metadata={
                "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
                "num_cases": len(case_results),
            },
        )
