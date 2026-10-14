
import re
import html
from typing import Optional


class TranscriptCleaner:
    """
    Cleans diarized call transcripts.

    - Keeps only CSR and CUSTOMER conversations
    - Removes IVR and other speaker segments
    - Masks common PII
    - Preserves conversation order and business context
    """

    ALLOWED_SPEAKERS = {"CSR", "CUSTOMER"}

    SPEAKER_PATTERN = re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9_ -]*):\s*(.*)$"
    )

    def __init__(self):
        self.pii_patterns = [
            (
                "EMAIL_ADDRESS",
                re.compile(
                    r"\b[A-Z0-9._%+-]+"
                    r"\s*@\s*"
                    r"[A-Z0-9.-]+\.[A-Z]{2,}\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "PHONE_NUMBER",
                re.compile(
                    r"(?<!\w)(?:\+?1[\s.-]?)?"
                    r"(?:\(\d{3}\)|\d{3})[\s.-]?"
                    r"\d{3}[\s.-]?\d{4}(?!\w)"
                ),
            ),
            (
                "SERVICE_ADDRESS",
                re.compile(
                    r"\b\d{1,6}\s+"
                    r"(?:[A-Za-z0-9]+\s+){1,6}"
                    r"(?:Street|St|Road|Rd|Avenue|Ave|"
                    r"Trail|Trl|Drive|Dr|Lane|Ln|"
                    r"Boulevard|Blvd|Court|Ct|Way|"
                    r"Place|Pl)\b\.?",
                    re.IGNORECASE,
                ),
            ),
            (
                "ZIP_CODE",
                re.compile(r"\b\d{5}(?:-\d{4})?\b"),
            ),
        ]

        self.name_patterns = [
            re.compile(
                r"(?i)(?<=my name is )"
                r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}"
            ),
            re.compile(
                r"(?i)(?<=speaking to )"
                r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}"
            ),
        ]

    def normalize_text(self, transcript: str) -> str:
        """Normalize HTML and whitespace."""

        if not transcript:
            return ""

        text = html.unescape(transcript)

        text = re.sub(
            r"<br\s*/?>",
            "\n",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def filter_speakers(self, transcript: str) -> str:
        """
        Keep only CSR and CUSTOMER turns.
        Remove IVR and other speaker turns.
        """

        cleaned_lines = []
        current_speaker = None

        for line in transcript.splitlines():
            line = line.strip()

            if not line:
                continue

            match = self.SPEAKER_PATTERN.match(line)

            if match:
                speaker = match.group(1).strip().upper()
                content = match.group(2).strip()

                current_speaker = speaker

                if speaker in self.ALLOWED_SPEAKERS:
                    if content:
                        cleaned_lines.append(
                            f"{speaker}: {content}"
                        )

            elif (
                current_speaker in self.ALLOWED_SPEAKERS
                and cleaned_lines
            ):
                cleaned_lines[-1] += " " + line

        return "\n".join(cleaned_lines)

    def remove_pii(self, transcript: str) -> str:
        """Mask common personally identifiable information."""

        text = transcript

        for label, pattern in self.pii_patterns:
            text = pattern.sub(f"[{label}]", text)

        for pattern in self.name_patterns:
            text = pattern.sub("[PERSON_NAME]", text)

        return text

    def clean(self, transcript: Optional[str]) -> str:
        """Main cleaning pipeline."""

        if not transcript:
            return ""

        # Step 1: Normalize transcript
        text = self.normalize_text(transcript)

        # Step 2: Remove IVR and other speaker turns
        text = self.filter_speakers(text)

        # Step 3: Mask PII
        text = self.remove_pii(text)

        # Step 4: Final whitespace cleanup
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()


def clean_transcript(transcript: str) -> str:
    """Public helper for existing pipeline integration."""

    cleaner = TranscriptCleaner()
    return cleaner.clean(transcript)
