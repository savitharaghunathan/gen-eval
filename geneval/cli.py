import sys

import click

from geneval.exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError
from geneval.framework import GenEvalFramework
from geneval.profile_manager import ProfileManager


@click.group()
def cli():
    pass


@cli.command()
@click.option("--profile", default=None, help="Profile name to evaluate with")
@click.option("--policy", default=None, help="Policy name to evaluate with")
@click.option("--data", "data_path", required=True, help="Path to test data YAML")
@click.option("--config", "config_path", default="./config/llm_config.yaml", help="Path to LLM config YAML")
@click.option("--profiles", "profiles_path", default=None, help="Path to profiles YAML")
@click.option("--output", "output_path", default=None, help="Write JSON results to file")
@click.option("--format", "output_format", type=click.Choice(["json", "table"]), default="json", help="Output format")
def evaluate(profile, policy, data_path, config_path, profiles_path, output_path, output_format):
    if not profile and not policy:
        raise click.UsageError("Either --profile or --policy is required")
    if profile and policy:
        raise click.UsageError("--profile and --policy are mutually exclusive")

    try:
        framework = GenEvalFramework(config_path=config_path)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    try:
        result = framework.evaluate_profile_batch(
            data_path=data_path,
            profile=profile,
            policy=policy,
            profiles_path=profiles_path,
        )
    except (ProfileValidationError, UnknownMetricError, ProfileNotFoundError, FileNotFoundError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    if output_format == "json":
        json_str = result.model_dump_json(indent=2)
        if output_path:
            with open(output_path, "w") as f:
                f.write(json_str)
            click.echo(f"Results written to {output_path}")
        else:
            click.echo(json_str)
    elif output_format == "table":
        _print_table(result)

    sys.exit(0 if result.overall_passed else 1)


def _print_table(result):
    click.echo(f"\nProfile: {result.profile_name}")
    if result.policy_name:
        click.echo(f"Policy: {result.policy_name}")
    click.echo(f"Pass Rate: {result.pass_rate:.0%} ({sum(1 for c in result.case_results if c.overall_passed)}/{len(result.case_results)})")
    click.echo()

    click.echo(f"{'Metric':<30} {'Mean Score':<12} {'Threshold':<12}")
    click.echo("-" * 54)
    for metric_name, stats in result.summary.items():
        mean = stats.get("mean", 0.0)
        threshold = ""
        if result.case_results:
            for mr in result.case_results[0].metric_results:
                if mr.name == metric_name:
                    threshold = f"{mr.threshold:.2f}"
                    break
        click.echo(f"{metric_name:<30} {mean:<12.4f} {threshold:<12}")

    click.echo()
    status = "PASSED" if result.overall_passed else "FAILED"
    click.echo(f"Overall: {status}")


@cli.group()
def profiles():
    pass


@profiles.command("list")
@click.option("--profiles", "profiles_path", default=None, help="Path to profiles YAML")
def profiles_list(profiles_path):
    try:
        pm = ProfileManager(profiles_path=profiles_path)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    for name in sorted(pm.list_profiles()):
        profile = pm.get_profile(name)
        desc = profile.get("description", "")
        click.echo(f"  {name:<25} {desc}")


@profiles.command("show")
@click.argument("name")
@click.option("--profiles", "profiles_path", default=None, help="Path to profiles YAML")
def profiles_show(name, profiles_path):
    try:
        pm = ProfileManager(profiles_path=profiles_path)
        profile = pm.get_profile(name)
    except (FileNotFoundError, ProfileNotFoundError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)

    click.echo(f"\nProfile: {name}")
    click.echo(f"Description: {profile.get('description', 'N/A')}")
    click.echo(f"Metrics: {', '.join(profile['metrics'])}")
    click.echo(f"Composite Threshold: {profile.get('composite_threshold', 'N/A')}")
    click.echo()
    click.echo(f"{'Metric':<30} {'Weight':<10} {'Threshold':<10}")
    click.echo("-" * 50)
    for m in profile["metrics"]:
        w = profile["weights"].get(m, 0)
        c = profile["criteria"].get(m, 0)
        click.echo(f"{m:<30} {w:<10.2f} {c:<10.2f}")


@profiles.command("validate")
@click.argument("path")
def profiles_validate(path):
    try:
        ProfileManager(profiles_path=path)
        click.echo(f"Valid: {path}")
    except (ProfileValidationError, UnknownMetricError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
