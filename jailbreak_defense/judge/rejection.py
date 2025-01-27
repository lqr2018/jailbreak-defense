from .constant import REJECTION_PREFIXES, EXCLUDED_KEYWORDS


def check_rejection(response, exclude_lack_of_info=True):
    # check whether the response is rejecting the input prompt
    # return whether the model rejects to response
    rejection = any([
        prefix.lower() in response.lower() for prefix in REJECTION_PREFIXES
    ])

    if exclude_lack_of_info:
        rejection = rejection and not any([
            keyword.lower() in response.lower() for keyword in EXCLUDED_KEYWORDS
        ])

    return rejection