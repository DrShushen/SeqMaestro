import json
import logging
from copy import copy

from deft.code_guard import compile_feature_function
from deft.provenance import ORIGIN_LLM
from deft.utils.feature import FeatureInfo

logger = logging.getLogger(__name__)


def _parse_items_for_prompt(prompt_items, parse_item_fn):
    """Parse all generations for one prompt with shared append-on-success logic."""
    parsed_items = []
    for prompt_item in prompt_items:
        parsed_item = parse_item_fn(prompt_item)
        if parsed_item is not None:
            parsed_items.append(parsed_item)
    return parsed_items


def extract_generations(list_responses):
    """Extract generations."""
    list_generations_all_prompts = []
    indices_extracted = []
    total_token_used = 0

    for i, response in enumerate(list_responses):
        if response is not None:
            # Update token usage
            token_used = response["usage"]["total_tokens"]
            total_token_used += token_used

            try:
                # Parse the equations
                list_generations = parsing_end2end(response)
                list_generations_all_prompts.append(list_generations)
                indices_extracted.append(i)

            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
    assert len(list_generations_all_prompts) == len(indices_extracted), (
        f"Extracted {len(list_generations_all_prompts)} generation groups but "
        f"{len(indices_extracted)} prompt indices; they are zipped together, so a mismatch "
        f"would attach a generation to the wrong prompt."
    )
    return list_generations_all_prompts, indices_extracted, total_token_used


def parse_code_feature(list_responses, list_features):
    # We use the list_features to add the function at the end of the FeatureInfo object
    """Parse code feature."""
    assert len(list_responses) == len(list_features), (
        f"Got {len(list_responses)} responses for {len(list_features)} features; they are "
        f"paired by index, so a mismatch would attach code to the wrong proposal."
    )

    full_list_code_str, indices_extracted, total_token_used = extract_generations(
        list_responses
    )
    list_feature_all_prompts = []

    for list_code, prompt_index in zip(full_list_code_str, indices_extracted, strict=True):
        source_feature = list_features[prompt_index]

        def parse_code_item(code_str, source_feature=source_feature):
            """Parse code item."""
            cleaned_code = code_str

            try:
                # Inside the guard, not before it. A completion that carries no `def` at all is
                # just one more way for a candidate to be unusable, and the surrounding design
                # -- returning None so `_parse_items_for_prompt` drops it -- already covers that
                # case. Raising instead lost the whole tree fit, hours in, to a single malformed
                # completion out of thousands.
                cleaned_code = clean_code_string(code_str)
                converted_fn = convert_code_to_fn(cleaned_code)
                updated_feature = copy(source_feature)
                updated_feature.fn = converted_fn
                updated_feature.string = cleaned_code
                return updated_feature
            except Exception as e:
                logger.error(f"Failed to convert the code to a Python function: {e}")
                logger.error(f"The clean code was: {cleaned_code}")
                logger.info(f"The code is: {code_str}")
                return None

        list_feature = _parse_items_for_prompt(list_code, parse_code_item)
        list_feature_all_prompts.append(list_feature)

    return list_feature_all_prompts, total_token_used


def clean_code_string(code_str):
    """Clean the code string by extracting everything from the first 'def' keyword.
    """
    if "def " not in code_str:
        raise ValueError("No function definition found in code string")

    # Extract everything from the first 'def'
    code = code_str[code_str.index("def ") :]

    # Remove any trailing backticks or whitespace
    code = code.strip("`").strip()

    return code


def parse_proposed_feature(list_responses, output_rationale=True):
    """Parse the LLM reponses obtained via an async call.
    """
    full_list_features, _, total_token_used = extract_generations(list_responses)
    # A list of lists: one entry per prompt, each holding that prompt's generations.

    list_converted_features_all_prompts = []
    for list_proposed_feature_str in full_list_features:

        def parse_proposed_item(proposed_feature_str):
            """Parse proposed item."""
            logger.info(f"Proposed feature: {proposed_feature_str}")
            cleaned_json_str = clean_json_string(proposed_feature_str)

            try:
                # Convert the json to an actual dictionary
                converted_dic = json.loads(cleaned_json_str)

                logger.info(f"Converted dictionary: {converted_dic}")
                # Create a FeatureInfo object
                if output_rationale:
                    converted_feature = FeatureInfo(
                        fn=None,
                        name=converted_dic["name"],
                        description=converted_dic["description"],
                        rationale=converted_dic["rationale"],
                        origin=ORIGIN_LLM,
                    )
                else:
                    # If no rationale is provided, we can still create the FeatureInfo object
                    # but we will not include the rationale field
                    # This is useful for cases where the LLM does not provide a rationale
                    converted_feature = FeatureInfo(
                        fn=None,
                        name=converted_dic["name"],
                        description=converted_dic["description"],
                        rationale="",
                        origin=ORIGIN_LLM,
                    )

                return converted_feature
            except Exception as e:
                logger.error(f"Failed to convert the proposed feature: {e}")
                return None

        list_converted_features = _parse_items_for_prompt(
            list_proposed_feature_str, parse_proposed_item
        )
        list_converted_features_all_prompts.append(list_converted_features)

    return list_converted_features_all_prompts, total_token_used


def clean_json_string(json_str):
    """Clean the JSON string by extracting between the first '{' and last '}'.
    """
    if "{" not in json_str or "}" not in json_str:
        return json_str

    # Extract from the first '{' to the last '}'
    start = json_str.index("{")
    end = json_str.rindex("}") + 1  # +1 to include the closing brace
    return json_str[start:end]


def parsing_end2end(response):
    """Handle parsing end2end."""
    list_generations = []
    n_processes = len(response["choices"])

    for process_i in range(n_processes):

        data = response["choices"][process_i]["message"]["content"]
        if data.startswith("##"):
            data = data[2:]
        if data.endswith("##"):
            data = data[:-2]

        list_generations.append(data)

    return list_generations


def convert_code_to_fn(code_str):
    """Compile a string of generated Python into a callable feature function.

    The guardrails live in `src/deft/code_guard.py`: what replaced the substring scan for
    `"exec"`/`"eval"` that used to stand here, which `__import__("os")` walked straight through
    and which also rejected good code containing `eval` in a comment.
    """
    return compile_feature_function(code_str)


def parse_repaired_feature(response_text, original):
    """A reply to a repair prompt, as a candidate, or ValueError saying why it cannot be one.

    The reply is a JSON object with the feature's name, description and code
    (`params_generation.instructions_code_repair`). A correction that keeps the name keeps the
    original's description where it gives none. One under a new name must describe what it now
    computes, and records the name it replaces. The code goes through the guard like any other.
    """
    try:
        payload = json.loads(clean_json_string(response_text or ""))
    except json.JSONDecodeError as error:
        raise ValueError(f"the reply was not a JSON object: {error}") from error
    if not isinstance(payload, dict) or not payload.get("code"):
        raise ValueError("the reply gave no code")
    name = str(payload.get("name") or original.name).strip()
    description = str(payload.get("description") or "").strip()
    renamed = name != original.name
    if renamed and not description:
        raise ValueError(f"the correction was given a new name, {name!r}, with no description of "
                         f"what it now computes")
    code = clean_code_string(str(payload["code"]))
    # The name first proposed, where `original` is itself an earlier correction.
    proposed = original.renamed_from or original.name
    return FeatureInfo(
        fn=convert_code_to_fn(code),
        name=name,
        description=description or original.description,
        string=code,
        rationale=original.rationale,
        origin=ORIGIN_LLM,
        repaired=True,
        renamed_from=proposed if name != proposed else None,
    )
