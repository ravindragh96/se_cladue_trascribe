"""
==============================================================================
Track A - Revenue Opportunity Detection

Detector Module

Responsibilities
----------------
1. Build prompts
2. Call LLMClient
3. Parse JSON
4. Validate output
5. Apply business rules
6. Return standardized business response

This module contains NO Databricks-specific logic.

Author:
    Ravindra Gantasala
==============================================================================
"""

import json
import logging
import re
from copy import deepcopy
from typing import Any, Dict, List

from modules.llm_client import LLMClient

logger = logging.getLogger(__name__)

VALID_TRADES = {
    "hvac",
    "plumbing",
    "electrical",
    "none"
}

VALID_SERVICE_CATEGORIES = {
    "repair",
    "maintenance",
    "installation",
    "inspection",
    "replacement",
    "tune-up",
    "iaq",
    "drain",
    "water heater",
    "none"
}

VALID_URGENCY = {
    "emergency",
    "urgent",
    "same_day",
    "prompt",
    "routine",
    "none"
}

VALID_APPOINTMENT_STATUS = {
    "requested",
    "scheduled",
    "not_requested",
    "not_scheduled",
    "transferred",
    "unknown"
}

VALID_CONFIDENCE = {
    "high",
    "medium",
    "low"
}

DEFAULT_RESPONSE = {
    "core_intent": "",
    "customer_problem": "",
    "customer_goal": "",
    "requested_service": "",
    "trade": "none",
    "service_category": "none",
    "urgency": "none",
    "appointment_status": "not_scheduled",
    "pricing_discussed": False,
    "revenue_opportunity": False,
    "intent_strength": 1,
    "confidence": "low",
    "confidence_score": 0,
    "evidence": [],
    "business_summary": "",
    "business_reason": ""
}


def default_response() -> Dict[str, Any]:
    """
    Returns a deep copy of default response.
    """
    return deepcopy(DEFAULT_RESPONSE)

# ==============================================================================
# Prompt Manager
# ==============================================================================

class PromptManager:
    """
    Builds prompts for the Revenue Opportunity Detection model.
    """

    @staticmethod
    def system_prompt() -> str:
        from pathlib import Path
        path = Path(__file__).resolve().with_name("prompt.txt")
        return path.read_text(encoding="utf-8-sig").strip()

    @staticmethod
    def user_prompt(transcript: str) -> str:

        transcript = transcript.strip()

        return f"""
Analyze the following customer service transcript according to the
revenue opportunity classification rules provided in the system prompt.

Determine the customer's PRIMARY intent and whether the customer expressed
a NEW or POTENTIAL revenue opportunity.

Pay particular attention to whether the customer is discussing an EXISTING
appointment/job versus requesting a NEW or ADDITIONAL service.

Transcript:

----------
{transcript}
----------

Return ONLY the JSON object.
"""
    
# ==============================================================================
# JSON Parser
# ==============================================================================

class JsonParser:
    """
    Handles extraction and parsing of JSON returned by the LLM.
    """

    @staticmethod
    def clean_response(response: str) -> str:
        """
        Removes markdown wrappers and unwanted tags.
        """

        if not response:
            return ""

        response = response.strip()

        response = re.sub(
            r"^```json",
            "",
            response,
            flags=re.IGNORECASE
        )

        response = re.sub(
            r"^```",
            "",
            response
        )

        response = re.sub(
            r"```$",
            "",
            response
        )

        response = response.replace("<think>", "")
        response = response.replace("</think>", "")

        return response.strip()

    @staticmethod
    def extract_json(text: str) -> str:
        """
        Extract the first valid JSON object from text.
        """

        if not text:
            raise ValueError("Empty model response.")

        text = JsonParser.clean_response(text)

        start = text.find("{")

        if start == -1:
            raise ValueError("JSON start '{' not found.")

        brace_count = 0

        for index in range(start, len(text)):

            if text[index] == "{":
                brace_count += 1

            elif text[index] == "}":
                brace_count -= 1

                if brace_count == 0:
                    return text[start:index + 1]

        raise ValueError("Incomplete JSON object.")

    @staticmethod
    def parse(text: str) -> Dict[str, Any]:
        """
        Returns parsed JSON dictionary.
        """

        json_text = JsonParser.extract_json(text)

        try:
            return json.loads(json_text)

        except json.JSONDecodeError as e:

            logger.exception("JSON decoding failed.")

            raise ValueError(
                f"Invalid JSON returned by model: {e}"
            )

    @staticmethod
    def safe_parse(text: str) -> Dict[str, Any]:
        """
        Safe JSON parser.
        Returns default response on failure.
        """

        try:

            return JsonParser.parse(text)

        except Exception as e:

            logger.exception(e)

            result = default_response()

            result["business_reason"] = (
                f"JSON parsing failed : {str(e)}"
            )

            return result
        

# ==============================================================================
# Validator
# ==============================================================================

class Validator:
    """
    Validates and normalizes LLM output.
    """

    @staticmethod
    def normalize_text(value: Any) -> str:

        if value is None:
            return ""

        return str(value).strip()

    @staticmethod
    def normalize_boolean(value: Any) -> bool:

        if isinstance(value, bool):
            return value

        if isinstance(value, str):

            value = value.strip().lower()

            return value in {
                "true",
                "yes",
                "1",
                "y"
            }

        return bool(value)

    @staticmethod
    def normalize_integer(
        value: Any,
        minimum: int = 1,
        maximum: int = 10
    ) -> int:

        try:

            value = int(value)

        except Exception:

            value = minimum

        value = max(minimum, min(maximum, value))

        return value

    @staticmethod
    def normalize_trade(value: Any) -> str:

        value = Validator.normalize_text(value).lower()

        if value in VALID_TRADES:
            return value

        return "none"

    @staticmethod
    def normalize_service_category(value: Any) -> str:

        value = Validator.normalize_text(value).lower()

        aliases = {

            "service": "repair",

            "repair service": "repair",

            "diagnostic": "repair",

            "diagnosis": "repair",

            "install": "installation",

            "replace": "replacement",

            "maintenance visit": "maintenance",

            "tune up": "tune-up",

            "tuneup": "tune-up",

            "air quality": "iaq",

            "indoor air quality": "iaq",

            "waterheater": "water heater",

            "drain cleaning": "drain"

        }

        value = aliases.get(value, value)

        if value in VALID_SERVICE_CATEGORIES:
            return value

        return "none"

    @staticmethod
    def normalize_urgency(value: Any) -> str:

        value = Validator.normalize_text(value).lower()

        aliases = {

            "urgent": "same_day",

            "today": "same_day",

            "immediate": "emergency",

            "asap": "same_day",

            "normal": "routine"

        }

        value = aliases.get(value, value)

        if value in VALID_URGENCY:
            return value

        return "none"

    @staticmethod
    def normalize_appointment(value: Any) -> str:

        value = Validator.normalize_text(value).lower()

        aliases = {

            "booked": "scheduled",

            "booking": "requested",

            "appointment requested": "requested",

            "appointment scheduled": "scheduled",

            "not booked": "not_scheduled",

            "none": "not_scheduled"

        }

        value = aliases.get(value, value)

        if value in VALID_APPOINTMENT_STATUS:
            return value

        return "not_scheduled"

    @staticmethod
    def normalize_confidence(value: Any) -> str:

        value = Validator.normalize_text(value).lower()

        if value in VALID_CONFIDENCE:
            return value

        return "low"

    @staticmethod
    def normalize_evidence(value: Any) -> List[str]:

        if value is None:
            return []

        if isinstance(value, list):

            return [
                str(x).strip()
                for x in value
                if str(x).strip()
            ]

        if isinstance(value, str):

            value = value.strip()

            if value:

                return [value]

        return []

    @staticmethod
    def validate(result: Dict[str, Any]) -> Dict[str, Any]:

        output = default_response()

        # Preserve the model's explicit opportunity classification.
        raw_revenue = result.get("revenue_opportunity")
        if isinstance(raw_revenue, bool):
            output["revenue_opportunity"] = raw_revenue
        elif isinstance(raw_revenue, str) and raw_revenue.strip().lower() in {"true", "false"}:
            output["revenue_opportunity"] = raw_revenue.strip().lower() == "true"
        else:
            raise ValueError("Missing/invalid revenue_opportunity decision in model output")

        output["core_intent"] = Validator.normalize_text(
            result.get("core_intent")
        )

        output["customer_problem"] = Validator.normalize_text(
            result.get("customer_problem")
        )

        output["customer_goal"] = Validator.normalize_text(
            result.get("customer_goal")
        )

        output["requested_service"] = Validator.normalize_text(
            result.get("requested_service")
        )

        output["trade"] = Validator.normalize_trade(
            result.get("trade")
        )

        output["service_category"] = Validator.normalize_service_category(
            result.get("service_category")
        )

        output["urgency"] = Validator.normalize_urgency(
            result.get("urgency")
        )

        output["appointment_status"] = Validator.normalize_appointment(
            result.get("appointment_status")
        )

        output["pricing_discussed"] = Validator.normalize_boolean(
            result.get("pricing_discussed")
        )

        output["intent_strength"] = Validator.normalize_integer(
            result.get("intent_strength"),
            minimum=1,
            maximum=10
        )

        output["confidence"] = Validator.normalize_confidence(
            result.get("confidence")
        )

        output["business_summary"] = Validator.normalize_text(
            result.get("business_summary")
        )

        output["business_reason"] = Validator.normalize_text(
            result.get("business_reason")
        )

        output["evidence"] = Validator.normalize_evidence(
            result.get("evidence")
        )

        return output
# ==============================================================================
# Business Rules
# ==============================================================================

class BusinessRules:
    """
    Applies business logic after validation.
    """

    @staticmethod
    def compute_revenue_opportunity(result: Dict[str, Any]) -> bool:
        # Use the transcript-based model decision. Do not infer revenue from
        # trade, requested_service, appointment status or intent strength.
        return result["revenue_opportunity"]

    # ------------------------------------------------------------------

    @staticmethod
    def compute_confidence_score(result: Dict[str, Any]) -> int:

        score = 0

        if result["core_intent"]:
            score += 15

        if result["customer_problem"]:
            score += 15

        if result["customer_goal"]:
            score += 10

        if result["requested_service"]:
            score += 15

        if result["trade"] != "none":
            score += 10

        if result["service_category"] != "none":
            score += 10

        if result["appointment_status"] in {
            "requested",
            "scheduled",
            "transferred"
        }:
            score += 10

        if result["pricing_discussed"]:
            score += 5

        if result["evidence"]:
            score += 10

        score += result["intent_strength"]

        return min(score, 100)

    # ------------------------------------------------------------------

    @staticmethod
    def compute_confidence(score: int) -> str:

        if score >= 80:
            return "high"

        if score >= 55:
            return "medium"

        return "low"

    # ------------------------------------------------------------------

    @staticmethod
    def generate_business_summary(result: Dict[str, Any]) -> str:

        if result["business_summary"]:
            return result["business_summary"]

        summary = []

        if result["customer_problem"]:
            summary.append(result["customer_problem"])

        if result["requested_service"]:
            summary.append(
                f"Requested {result['requested_service']}"
            )

        elif result["service_category"] != "none":
            summary.append(
                f"Requested {result['service_category']} service"
            )

        if result["appointment_status"] in {
            "requested",
            "scheduled"
        }:
            summary.append("Appointment requested")

        return ". ".join(summary)

    # ------------------------------------------------------------------

    @staticmethod
    def generate_business_reason(result: Dict[str, Any]) -> str:

        if result["business_reason"]:
            return result["business_reason"]

        if result["revenue_opportunity"]:

            reasons = []

            if result["requested_service"]:
                reasons.append(
                    "Customer explicitly requested service."
                )

            if result["trade"] != "none":
                reasons.append(
                    f"Trade identified as {result['trade']}."
                )

            if result["service_category"] != "none":
                reasons.append(
                    f"Service category is {result['service_category']}."
                )

            if result["appointment_status"] in {
                "requested",
                "scheduled",
                "transferred"
            }:
                reasons.append(
                    "Customer attempted to schedule an appointment."
                )

            return " ".join(reasons)

        return (
            "Transcript does not contain sufficient evidence "
            "of a customer revenue opportunity."
        )

    # ------------------------------------------------------------------

    @staticmethod
    def apply(result: Dict[str, Any]) -> Dict[str, Any]:

        result["revenue_opportunity"] = (
            BusinessRules.compute_revenue_opportunity(result)
        )

        result["confidence_score"] = (
            BusinessRules.compute_confidence_score(result)
        )

        result["confidence"] = (
            BusinessRules.compute_confidence(
                result["confidence_score"]
            )
        )

        result["business_summary"] = (
            BusinessRules.generate_business_summary(result)
        )

        result["business_reason"] = (
            BusinessRules.generate_business_reason(result)
        )

        return result
    
# ==============================================================================
# Revenue Opportunity Analyzer
# ==============================================================================

class RevenueOpportunityAnalyzer:
    """
    Main detector pipeline.

    Pipeline
    --------
    Transcript
        ↓
    PromptManager
        ↓
    Claude
        ↓
    JSON Parser
        ↓
    Validator
        ↓
    Business Rules
        ↓
    Final Output
    """

    def __init__(self, llm: LLMClient):

        self.llm = llm

    # ------------------------------------------------------------------

    def analyze(
        self,
        transcript: str
    ) -> Dict[str, Any]:

        logger.info("=" * 80)
        logger.info("Revenue Opportunity Analysis Started")
        logger.info("=" * 80)

        if transcript is None:

            result = default_response()

            result["business_reason"] = "Transcript is None."

            return result

        transcript = str(transcript).strip()

        if transcript.lower() in {"nan", "none", "null"}:
            transcript = ""

        if not transcript:

            result = default_response()

            result["business_reason"] = "Transcript is empty."

            return result

        try:

            logger.info("Generating LLM response...")

            response = self.llm.generate(

                system_prompt=PromptManager.system_prompt(),

                user_prompt=PromptManager.user_prompt(transcript),

                temperature=0.0,

                max_tokens=1500

            )

            logger.info("LLM response received.")

        except Exception as e:

            logger.exception(e)

            result = default_response()

            result["business_reason"] = (
                f"PROCESSING_ERROR: LLM inference failed ({type(e).__name__})."
            )

            return result

        # --------------------------------------------------------------
        # JSON Parsing
        # --------------------------------------------------------------

        logger.info("Parsing JSON response...")

        try:
            parsed = JsonParser.parse(response)
        except (ValueError, TypeError) as exc:
            result = default_response()
            result["business_reason"] = f"PROCESSING_ERROR: Model returned invalid JSON ({type(exc).__name__})."
            return result

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        logger.info("Validating extracted fields...")

        try:
            validated = Validator.validate(parsed)
        except (ValueError, TypeError) as exc:
            result = default_response()
            result["business_reason"] = f"PROCESSING_ERROR: Missing or invalid model classification ({type(exc).__name__})."
            return result

        # --------------------------------------------------------------
        # Business Rules
        # --------------------------------------------------------------

        logger.info("Applying business rules...")

        final_output = BusinessRules.apply(validated)

        logger.info("=" * 80)
        logger.info("Revenue Opportunity Analysis Completed")
        logger.info("=" * 80)

        return final_output

    # ------------------------------------------------------------------

    def __call__(
        self,
        transcript: str
    ) -> Dict[str, Any]:

        return self.analyze(transcript)
    
    # def create_detector(llm: LLMClient) -> RevenueOpportunityAnalyzer:
    #     """
    #     Factory method for creating the detector.
    #     """

    #     return RevenueOpportunityAnalyzer(llm)
