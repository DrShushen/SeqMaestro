import pandas as pd

from deft.utils.dataset import analyze_column_types


def construct_prompt_name_description(
    X,
    target_name,
    dataset_info,
    history,
    context_template,
    instructions,
    include_ic_samples,
    include_history,
    include_feature_description,
    use_description_in_history=True,
    n_ic_samples=2,
    rng=None,
    column_descriptions=None,
    max_sequence_chars_in_prompt=None,
):
    """Handle construct prompt name description."""
    prompt_context = context_template.format(
        target_name=target_name, dataset_info=dataset_info
    )

    prompt_feature_description = serialize_features(X, column_descriptions)

    prompt_in_context = serialize_in_context(
        X, n_ic_samples, rng=rng, max_chars=max_sequence_chars_in_prompt
    )

    prompt_history = serialize_history(
        history,
        include_history=include_history,
        use_description=use_description_in_history,
    )

    prompt = prompt_context
    if include_feature_description:
        prompt += "\n\n" + prompt_feature_description
    if include_ic_samples:
        prompt += "\n\n" + prompt_in_context

    prompt += "\n\n" + prompt_history

    prompt += "\n\n" + instructions

    return prompt


def construct_prompt_reflection(
    X,
    list_features,
    target_name,
    dataset_info,
    history,
    context_template,
    instructions,
    include_ic_samples,
    include_history,
    include_feature_description,
    sort_by_score=True,
    sort_best_first=False,
    use_description_in_history=True,
    n_ic_samples=2,
    include_code=False,
    include_rationale=False,
    selector=None,
    n_parents_per_prompt=1,
    rng=None,
    column_descriptions=None,
    max_sequence_chars_in_prompt=None,
):
    """Handle construct prompt reflection."""
    list_parents = selector.select(list_features, n_parents_per_prompt)

    prompt_context = context_template.format(
        target_name=target_name, dataset_info=dataset_info
    )  # "This is a dataset about RNA polymerase, your task is to ..."

    prompt_feature_description = serialize_features(X, column_descriptions)

    prompt_in_context = serialize_in_context(
        X, n_ic_samples, rng=rng, max_chars=max_sequence_chars_in_prompt
    )

    # `use_description_in_history` used to stop at the initial prompt: this call passed two
    # positional arguments and did not accept the third, so setting the config key to false
    # silently changed one prompt of the two.
    prompt_history = serialize_history(
        history,
        include_history=include_history,
        use_description=use_description_in_history,
    )

    prompt_population = serialize_list_features(
        list_parents,
        sort_by_score=sort_by_score,
        sort_best_first=sort_best_first,
        include_code=include_code,
        include_rationale=include_rationale,
    )

    prompt = prompt_context
    if include_feature_description:
        prompt += "\n\n" + prompt_feature_description
    if include_ic_samples:
        prompt += "\n\n" + prompt_in_context

    prompt += "\n\n" + instructions

    prompt += "\n\n" + prompt_population

    prompt += "\n\n" + prompt_history

    return prompt


def serialize_list_features(
    list_features,
    sort_by_score=True,
    sort_best_first=False,
    include_code=False,
    include_rationale=False,
):
    """Render a population of candidate features for a reflection prompt.

    `sort_best_first` selects the direction and defaults to False -- worst first, best last --
    because that is what this has always done, and the pilot's prompts must not change under it.
    It reads as a mistake next to every other sort in the codebase, which is descending, so it
    is now a setting rather than a silent convention:
    `params_generation.sort_reflection_best_first`. Each feature is printed with its score
    either way, so neither order withholds anything.
    """
    prompt = "<Beginning of the population of features> \n"
    prompt += "Here is the list of features along with their score: \n"
    if sort_by_score:
        list_features = sorted(list_features, key=lambda x: x.score, reverse=sort_best_first)
    # Iterate through selected samples
    for i, feature in enumerate(list_features):
        # Start new sample
        prompt += f"Feature {i + 1} \n"
        prompt += (
            f"Score: {feature.score:.4f} \n Feature name: {feature.name} \n "
            f"Feature description: {feature.description}"
        )
        if include_rationale:
            prompt += f"\n  Rationale: {feature.rationale}"
        if include_code:
            prompt += f"\n Feature code: {feature.string}"



        # Add a new line
        prompt += "\n"

        # Remove trailing comma and add newline
        prompt = prompt.rstrip(",")

    prompt += "<End of the population of features>"

    return prompt.rstrip()


def serialize_in_context(X, n_samples, rng=None, max_chars=None):
    """
    Constructs a prompt with randomly selected samples from a DataFrame.

    A text value longer than `max_chars` is shown by its first and last characters with its
    length stated (`shown_text`), since a whole transcript would fill the prompt. None shows every
    value whole.
    """
    # Randomly select n_samples rows from X
    # `rng` is the run's generator, threaded down from `random_state`. Without it, the
    # examples shown to the model changed on every call, so two runs with the same seed
    # were not comparable. Passing the generator itself rather than a fixed integer keeps
    # successive calls varied while making the whole sequence reproducible.
    selected_samples = X.sample(n=min(n_samples, len(X)), random_state=rng)

    # Initialize prompt string
    prompt = "To help you, I am also providing some randomly sampled examples from the dataset: \n"

    # Iterate through selected samples
    for i, (_, row) in enumerate(selected_samples.iterrows(), 1):
        # Start new sample
        prompt += f"Sample {i}:"

        # Add feature-value pairs
        for feature_name, value in row.items():
            prompt += f" {feature_name}: {shown_text(value, max_chars)},"

        # Remove trailing comma and add newline
        prompt = prompt.rstrip(",") + "\n"

    # Add the size of the datasets
    prompt += f"\n The dataset has {len(X)} samples."

    return prompt.rstrip()


def shown_text(value, max_chars):
    """A value as a prompt shows it: whole, or where it is text longer than `max_chars`, its
    first and last characters with its length stated."""
    if max_chars is None or not isinstance(value, str) or len(value) <= max_chars:
        return value
    head = max_chars // 2
    return (f"{value[:head]}...{value[len(value) - (max_chars - head):]}  "
            f"({len(value):,} characters, the middle not shown)")


def _number(value):
    """A range endpoint as a prompt shows it: to four significant figures where it is fractional,
    since a float's full repr says nothing a model can use and costs tokens."""
    if isinstance(value, float) and not float(value).is_integer():
        return f"{value:.4g}"
    return value


# How many distinct values a column may have before the prompt stops listing them and gives a
# summary instead. Numeric columns get a much lower bound: a list of floats costs many tokens
# and says less than a range does. Both are generous for real categoricals -- nucleotides,
# strands, chromosomes.
_MAX_ENUMERATED_NUMERIC_VALUES = 12
_MAX_ENUMERATED_VALUES = 30


def serialize_features(X, column_descriptions=None):
    """
    Creates a description of each feature in the DataFrame including type and statistics.

    A column in `column_descriptions` (`{column: text}`, the dataset's own words for what it
    means) carries its text after its type and range. A text column whose values differ in length
    gives the range of lengths, which code reading it must allow for.
    """
    column_descriptions = column_descriptions or {}
    lines = []
    for column, line in _column_lines(X):
        text = column_descriptions.get(column)
        lines.append(f"{line}: {text}" if text else line)
    return "\n".join(lines)


def _column_lines(X):
    """Each column with its type and statistics, as `(column, line)`."""
    # Get column types using the provided function
    column_types = analyze_column_types(X)

    # Initialize the description string
    descriptions = []

    for column in X.columns:
        col_type = column_types[column]

        # Start with basic feature name and type
        description = f"{column}: {col_type}"

        # Add additional information based on column type
        if col_type == "continuous" or col_type == "discrete":
            min_val = X[column].min()
            max_val = X[column].max()
            description += f" (range: {_number(min_val)} to {_number(max_val)})"

        elif col_type == "categorical" or col_type == "boolean":
            # Missing values are dropped *before* sorting, not after. Upstream sorted first,
            # which raised `TypeError: '<' not supported between 'str' and 'float'` on any
            # column mixing strings with NaN -- unreachable while DEFT was handed a single
            # sequence column, and immediate once it is handed covariates.
            values = X[column].dropna().unique()

            if pd.api.types.is_numeric_dtype(X[column]):
                ordered = sorted(values)
                if len(ordered) > _MAX_ENUMERATED_NUMERIC_VALUES:
                    # `analyze_column_types` calls a numeric column categorical whenever
                    # fewer than 5% of its values are distinct, which a sparse continuous
                    # signal easily satisfies. Enumerating it dumped up to 175 raw floats
                    # into every prompt -- ~300 tokens per column, telling the model nothing
                    # a range does not. Describe it as what it is instead.
                    kind = (
                        "discrete"
                        if pd.api.types.is_integer_dtype(X[column])
                        else "continuous"
                    )
                    description = (
                        f"{column}: {kind} (range: {_number(ordered[0])} to "
                        f"{_number(ordered[-1])}, {len(ordered)} distinct values)"
                    )
                    descriptions.append((column, description))
                    continue
                unique_values = [str(val) for val in ordered]
            else:
                unique_values = sorted(str(val) for val in values)
                if len(unique_values) > _MAX_ENUMERATED_VALUES:
                    shown = ", ".join(unique_values[:_MAX_ENUMERATED_VALUES])
                    description += (
                        f" ({len(unique_values)} distinct values, e.g. {shown}, ...)"
                    )
                    descriptions.append((column, description))
                    continue

            description += f" (values: {', '.join(unique_values)})"

        elif col_type == "datetime":
            min_date = X[column].min()
            max_date = X[column].max()
            description += f" (range: {min_date} to {max_date})"

        elif col_type == "text":
            lengths = X[column].str.len()
            if lengths.nunique() > 1:
                description += (
                    f" (length {int(lengths.min()):,} to {int(lengths.max()):,} characters, "
                    f"varying by row)"
                )
            else:
                description += f" (average length: {lengths.mean():.1f} characters)"

        descriptions.append((column, description))

    return descriptions


def serialize_history(history, include_history, use_description=True):
    """
    Serializes the decision tree path history into a readable format.
    """
    if not history or not include_history:
        return (
            "<Beginning Splitting conditions from root to current node> \n This is the "
            "root node (no conditions applied). <End of Splitting conditions from root to "
            "current node>"
        )

    # Start with the header
    result = "<Beginning Splitting conditions from root to current node> \n"

    # Add each condition
    for feature, operator in history:
        # Format the condition
        condition = f"{feature.name} {operator} than {feature.threshold:.3f}"
        if use_description:
            condition += f" ({feature.description})"
        result += condition + "\n"

    # Add the end of the history
    result += "<End of Splitting conditions from root to current node>"

    return result.rstrip()


def serialize_proposed_feature(feature):
    """
    Serializes a proposed feature into a formatted string.
    """
    template = (
        "The feature you should generate has the following characteristics:\n"
        "Feature name: {name}\n"
        "Feature description: {description}"
    )

    return template.format(name=feature.name, description=feature.description)


def construct_prompt_code_generation(
    X, context, proposed_feature, instructions, column_descriptions=None
):
    """Handle construct prompt code generation."""
    prompt_proposed_feature = serialize_proposed_feature(proposed_feature)
    prompt_features = serialize_features(X, column_descriptions)

    prompt = (
        context
        + "\n\n You should build this feature using the following original features:\n"
        + prompt_features
        + "\n\n"
        + prompt_proposed_feature
        + "\n\n"
        + instructions
    )
    return prompt


def construct_prompt_code_repair(
    X, context, feature, detail, guidance, instructions, column_descriptions=None
):
    """The code-generation prompt again, for a feature whose code failed on this node's rows.

    It shows the feature as proposed, the code last tried for it, what went wrong (`detail`, in
    the words of `deft.features.helpers.screen_features`), and how to correct that kind of failure
    (`guidance`, from `params_generation.repair_guidance`). `instructions`
    (`params_generation.instructions_code_repair`) asks for the answer: the name, description and
    code as JSON, since a correction that computes something new needs a name and description for
    it.
    """
    return (
        context
        + "\n\n You should build this feature using the following original features:\n"
        + serialize_features(X, column_descriptions)
        + "\n\n"
        + serialize_proposed_feature(feature)
        + "\n\nThis code was written for it, and failed when it was run on the rows at this "
        + "node:\n"
        + feature.string
        + f"\n\nWhat went wrong: it {detail.rstrip('.')}.\n\n"
        + guidance
        + "\n\n"
        + instructions
    )
