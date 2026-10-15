"""Track A - Revenue Opportunity Detection.

Standardized detector module supporting the 16-field contract, external
prompt.txt, safe parsing, grounded evidence extraction, and deterministic
business rules.
"""

from copy import deepcopy
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List

from modules.llm_client import LLMClient

logger = logging.getLogger(__name__)

VALID_TRADES = {'hvac', 'plumbing', 'electrical', 'none'}

VALID_SERVICE_CATEGORIES = {
    'repair',
    'maintenance',
    'installation',
    'inspection',
    'replacement',
    'none',
}

VALID_URGENCY = {'emergency', 'same_day', 'prompt', 'routine', 'none'}

VALID_APPOINTMENT_STATUS = {
    'requested',
    'scheduled',
    'not_scheduled',
    'transferred',
}

VALID_CONFIDENCE = {'high', 'medium', 'low'}

DEFAULT_RESPONSE = {
    'core_intent': '',
    'customer_problem': '',
    'customer_goal': '',
    'requested_service': '',
    'trade': 'none',
    'service_category': 'none',
    'urgency': 'none',
    'appointment_status': 'not_scheduled',
    'pricing_discussed': False,
    'revenue_opportunity': False,
    'intent_strength': 1,
    'confidence': 'low',
    'confidence_score': 0,
    'evidence': [],
    'business_summary': '',
    'business_reason': '',
}


def default_response() -> Dict[str, Any]:
  return deepcopy(DEFAULT_RESPONSE)


class PromptManager:
  """Builds prompts for Revenue Opportunity Detection using external prompt.txt."""

  @staticmethod
  def system_prompt() -> str:
    prompt_path = Path(__file__).resolve().with_name('prompt.txt')
    if prompt_path.exists():
      return prompt_path.read_text(encoding='utf-8-sig').strip()
    return (
        'Analyze inbound customer call transcripts to detect revenue'
        ' opportunities. Return valid JSON matching the 16-field schema.'
    )

  @staticmethod
  def user_prompt(transcript: str) -> str:
    clean_text = transcript.strip() if transcript else ''
    return (
        'Analyze the following customer service transcript according to the'
        ' rules in the system instructions. Extract verbatim evidence quotes,'
        ' categorize the call, and return the complete JSON object.\n\n'
        'Transcript:\n'
        '----------\n'
        f'{clean_text}\n'
        '----------\n\n'
        'Return ONLY the valid JSON object.'
    )


class JsonParser:
  """Safely handles JSON parsing and tag cleanup."""

  @staticmethod
  def clean_response(response: str) -> str:
    if not response:
      return ''
    response = response.strip()
    response = re.sub(
        r'<think\b[^>]*>.*?</think\s*>', '', response, flags=re.S | re.I
    )
    response = re.sub(r'^```json', '', response, flags=re.I)
    response = re.sub(r'^```', '', response)
    response = re.sub(r'```$', '', response)
    return response.strip()

  @staticmethod
  def extract_json(text: str) -> str:
    if not text:
      raise ValueError('Empty model response.')

    cleaned = JsonParser.clean_response(text)
    start = cleaned.find('{')
    if start == -1:
      raise ValueError("JSON start '{' not found.")

    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', cleaned):
      try:
        obj, end = decoder.raw_decode(cleaned[match.start() :])
        if isinstance(obj, dict):
          return cleaned[match.start() : match.start() + end]
      except json.JSONDecodeError:
        continue

    raise ValueError('Incomplete or malformed JSON object in model response.')

  @staticmethod
  def safe_parse(text: str) -> Dict[str, Any]:
    try:
      json_str = JsonParser.extract_json(text)
      return json.loads(json_str)
    except Exception as exc:
      logger.warning('JSON parsing fallback triggered: %s', exc)
      res = default_response()
      res['business_reason'] = f'JSON parsing failed: {str(exc)}'
      return res


class Validator:
  """Validates fields, normalizes enums, and verifies verbatim evidence."""

  @staticmethod
  def normalize_text(val: Any) -> str:
    return str(val).strip() if val is not None else ''

  @staticmethod
  def normalize_boolean(val: Any) -> bool:
    if isinstance(val, bool):
      return val
    if isinstance(val, str):
      return val.strip().lower() in {'true', 'yes', '1', 'y'}
    return bool(val)

  @staticmethod
  def normalize_trade(val: Any) -> str:
    val = Validator.normalize_text(val).lower()
    return val if val in VALID_TRADES else 'none'

  @staticmethod
  def normalize_service_category(val: Any) -> str:
    val = Validator.normalize_text(val).lower()
    aliases = {
        'service': 'repair',
        'diagnostic': 'repair',
        'install': 'installation',
        'replace': 'replacement',
        'tune-up': 'maintenance',
        'tuneup': 'maintenance',
    }
    val = aliases.get(val, val)
    return val if val in VALID_SERVICE_CATEGORIES else 'none'

  @staticmethod
  def normalize_urgency(val: Any) -> str:
    val = Validator.normalize_text(val).lower()
    aliases = {'immediate': 'emergency', 'asap': 'prompt', 'urgent': 'prompt'}
    val = aliases.get(val, val)
    return val if val in VALID_URGENCY else 'none'

  @staticmethod
  def normalize_appointment(val: Any) -> str:
    val = Validator.normalize_text(val).lower().replace(' ', '_')
    aliases = {
        'booked': 'scheduled',
        'booking': 'requested',
        'appointment_requested': 'requested',
        'appointment_scheduled': 'scheduled',
        'none': 'not_scheduled',
    }
    val = aliases.get(val, val)
    return val if val in VALID_APPOINTMENT_STATUS else 'not_scheduled'

  @staticmethod
  def filter_verbatim_evidence(raw_evidence: Any, transcript: str) -> List[str]:
    """Ensures quotes exist in transcript; drops hallucinated quotes without raising errors."""
    if not raw_evidence or not transcript:
      return []

    items = []
    if isinstance(raw_evidence, list):
      items = [str(x).strip() for x in raw_evidence if str(x).strip()]
    elif isinstance(raw_evidence, dict):
      items = [str(v).strip() for v in raw_evidence.values() if str(v).strip()]
    elif isinstance(raw_evidence, str) and raw_evidence.strip():
      items = [raw_evidence.strip()]

    grounded = []
    for item in items:
      if item in transcript:
        grounded.append(item)
        continue
      pattern = r'\s+'.join(re.escape(word) for word in item.split())
      match = re.search(pattern, transcript, flags=re.IGNORECASE)
      if match:
        grounded.append(match.group(0))
      else:
        logger.debug("Evidence '%s' dropped because not in transcript.", item)

    return grounded

  @staticmethod
  def validate(result: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    output = default_response()

    for field in (
        'core_intent',
        'customer_problem',
        'customer_goal',
        'requested_service',
        'business_summary',
        'business_reason',
    ):
      output[field] = Validator.normalize_text(result.get(field))

    output['trade'] = Validator.normalize_trade(result.get('trade'))
    output['service_category'] = Validator.normalize_service_category(
        result.get('service_category')
    )
    output['urgency'] = Validator.normalize_urgency(result.get('urgency'))
    output['appointment_status'] = Validator.normalize_appointment(
        result.get('appointment_status')
    )
    output['pricing_discussed'] = Validator.normalize_boolean(
        result.get('pricing_discussed')
    )
    output['revenue_opportunity'] = Validator.normalize_boolean(
        result.get('revenue_opportunity')
    )

    try:
      score = int(result.get('intent_strength', 1))
      output['intent_strength'] = max(1, min(10, score))
    except (ValueError, TypeError):
      output['intent_strength'] = 1

    conf = Validator.normalize_text(result.get('confidence')).lower()
    output['confidence'] = conf if conf in VALID_CONFIDENCE else 'low'

    output['evidence'] = Validator.filter_verbatim_evidence(
        result.get('evidence'), transcript
    )
    return output


class BusinessRules:
  """Calculates confidence score and ensures business consistency."""

  @staticmethod
  def compute_confidence_score(result: Dict[str, Any]) -> int:
    score = 0
    if result.get('core_intent'):
      score += 15
    if result.get('customer_problem'):
      score += 15
    if result.get('customer_goal'):
      score += 10
    if result.get('requested_service'):
      score += 15
    if result.get('trade') != 'none':
      score += 10
    if result.get('service_category') != 'none':
      score += 10
    if result.get('appointment_status') in {'requested', 'scheduled'}:
      score += 10
    if result.get('pricing_discussed'):
      score += 5
    if result.get('evidence'):
      score += 10
    return min(100, score)

  @staticmethod
  def apply(result: Dict[str, Any]) -> Dict[str, Any]:
    score = BusinessRules.compute_confidence_score(result)
    result['confidence_score'] = score

    if not result.get('confidence') or result['confidence'] == 'low':
      if score >= 75:
        result['confidence'] = 'high'
      elif score >= 50:
        result['confidence'] = 'medium'
      else:
        result['confidence'] = 'low'

    if not result.get('business_summary'):
      parts = [
          result.get('core_intent'),
          result.get('customer_problem'),
          result.get('customer_goal'),
      ]
      result['business_summary'] = (
          '. '.join(p for p in parts if p).strip() + '.'
      )

    if not result.get('business_reason'):
      if result['revenue_opportunity']:
        result['business_reason'] = (
            f"Customer expressed a need for {result['trade']} "
            f"{result['service_category']} service."
        )
      else:
        result['business_reason'] = (
            'The inquiry concerns administrative, billing, or existing-account'
            ' matters without a new billable service request.'
        )

    return result


class RevenueOpportunityAnalyzer:
  """Main detector pipeline."""

  def __init__(self, llm: LLMClient):
    self.llm = llm

  def analyze(self, transcript: str) -> Dict[str, Any]:
    if transcript is None or not str(transcript).strip():
      res = default_response()
      res['business_reason'] = 'Transcript is empty or null.'
      return res

    clean_transcript = str(transcript).strip()

    try:
      response = self.llm.generate(
          system_prompt=PromptManager.system_prompt(),
          user_prompt=PromptManager.user_prompt(clean_transcript),
          temperature=0.0,
          max_tokens=1500,
      )
    except Exception as exc:
      logger.exception('LLM inference error: %s', exc)
      res = default_response()
      res['business_reason'] = f'LLM inference failed: {str(exc)}'
      return res

    parsed = JsonParser.safe_parse(response)
    validated = Validator.validate(parsed, clean_transcript)
    final_result = BusinessRules.apply(validated)
    return final_result

  __call__ = analyze
