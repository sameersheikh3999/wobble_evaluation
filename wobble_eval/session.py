"""The evaluated coaching session, embedded as a gzip+base64 blob."""
import base64, gzip, json, re

import base64, gzip, json

SESSION_B64 = (
    "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"
    "Zm3ZE5194qc7FsqkqYwA6OtH//y/GM9FsVBpAAA="
)

def load_embedded():
    return json.loads(gzip.decompress(base64.b64decode(SESSION_B64)).decode("utf-8"))



def load_session(path=None):
    """Load the embedded session, or any JSON file with the same shape."""
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return load_embedded()


def session_meta(session):
    return {
        "session_id": session.get("session_id", "?"),
        "language": session.get("language", "?"),
        "duration_min": round(float(session.get("ffprobe_duration_seconds",
                              session.get("db_duration_seconds", 0)) or 0) / 60, 1),
        "recorded": session.get("created_at", "?"),
    }


def turn_stats(transcript):
    turns = re.findall(r"\[(\d\d:\d\d)\]\s*([A-Za-z]+)\s*\(([A-Z]{2})\):", transcript)
    speakers, langs = {}, {}
    for _, who, lang in turns:
        speakers[who] = speakers.get(who, 0) + 1
        langs[lang] = langs.get(lang, 0) + 1
    return dict(n_turns=len(turns), speakers=speakers, languages=langs,
                chars=len(transcript), words=len(transcript.split()))
