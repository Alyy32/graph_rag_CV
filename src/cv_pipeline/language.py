from lingua import Language, LanguageDetectorBuilder


LANGUAGE_CODES = {
    Language.FRENCH: "fr",
    Language.ENGLISH: "en",
}


def detect_language(text):
    detector = (
        LanguageDetectorBuilder
        .from_languages(Language.FRENCH, Language.ENGLISH)
        .with_minimum_relative_distance(0.20)
        .build()
    )

    confidence_values = detector.compute_language_confidence_values(text)
    detected = detector.detect_language_of(text)
    enough_text = sum(character.isalpha() for character in text) >= 80

    if not enough_text:
        detected = None
        status = "insufficient_text"
    elif detected is None:
        status = "uncertain"
    else:
        status = "detected"

    return {
        "dominant_language": LANGUAGE_CODES.get(detected, "und"),
        "status": status,
        "candidate_languages": ["fr", "en"],
        "scores": {
            LANGUAGE_CODES[item.language]: round(item.value, 6)
            for item in confidence_values
        },
        "esco_matching_languages": ["fr", "en"],
    }
