"""Emotion detection module using Watson NLP Emotion Predict service."""

from typing import Any, Dict, Optional

import requests


WATSON_EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)

HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}

EMPTY_RESPONSE = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}


def _format_response(emotions: Dict[str, float]) -> Dict[str, Any]:
    """Return the emotion scores with the dominant emotion added."""
    emotion_scores = {
        "anger": emotions.get("anger", 0),
        "disgust": emotions.get("disgust", 0),
        "fear": emotions.get("fear", 0),
        "joy": emotions.get("joy", 0),
        "sadness": emotions.get("sadness", 0),
    }
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    return {
        "anger": emotion_scores["anger"],
        "disgust": emotion_scores["disgust"],
        "fear": emotion_scores["fear"],
        "joy": emotion_scores["joy"],
        "sadness": emotion_scores["sadness"],
        "dominant_emotion": dominant_emotion,
    }


def _fallback_emotion_detector(text_to_analyze: str) -> Dict[str, Any]:
    """Provide a small fallback only when the remote service is unavailable."""
    text = text_to_analyze.lower()
    keyword_scores = {
        "anger": int(any(word in text for word in ("mad", "angry", "furious", "rage"))),
        "disgust": int(any(word in text for word in ("disgust", "disgusted", "gross"))),
        "fear": int(any(word in text for word in ("afraid", "fear", "scared", "terrified"))),
        "joy": int(any(word in text for word in ("glad", "happy", "joy", "love"))),
        "sadness": int(any(word in text for word in ("sad", "unhappy", "depressed"))),
    }
    return _format_response(keyword_scores)


def emotion_detector(text_to_analyze: Optional[str]) -> Dict[str, Any]:
    """
    Analyze text and return scores for anger, disgust, fear, joy, sadness,
    and the dominant emotion.
    """
    if text_to_analyze is None or not text_to_analyze.strip():
        return EMPTY_RESPONSE.copy()

    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(
            WATSON_EMOTION_URL,
            headers=HEADERS,
            json=payload,
            timeout=20,
        )
    except requests.exceptions.RequestException:
        return _fallback_emotion_detector(text_to_analyze)

    if response.status_code == 400:
        return EMPTY_RESPONSE.copy()

    response.raise_for_status()
    response_data = response.json()
    emotions = response_data["emotionPredictions"][0]["emotion"]

    return _format_response(emotions)
