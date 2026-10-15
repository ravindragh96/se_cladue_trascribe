"""Service Experts 14-field client-contract detector with grounded evidence repair."""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

ENUMS = {
    'trade': {'hvac','plumbing','electrical','multiple','none'},
    'service_category': {'repair','replacement_install','maintenance','estimate_quote','diagnostic','none'},
    'urgency': {'emergency','prompt','flexible','none'},
    'caller_relationship': {'claims_new','claims_existing_or_member','unclear'},
    'call_regarding': {'new_service_need','existing_job_or_appt','billing_admin','other'},
    'booked_on_call': {'booked','committed_no_time','declined','not_discussed','transferred','call_dropped'},
    'refusal_reason': {'price','scheduling','competitor','quotes_only','service_area','not_accepting','other','na'},
    'confidence': {'high','medium','low'},
}
DEFAULTS = {
    'trade':'none','service_category':'none','urgency':'none',
    'caller_relationship':'unclear','call_regarding':'other',
    'agent_offered_booking':False,'booked_on_call':'not_discussed',
    'refusal_reason':'na','refusal_paraphrase':None,'evidence':{},
    'summary':'The caller\'s request is unclear. The call outcome cannot be established.',
    'revenue_opportunity':False,'intent_strength':1,'confidence':'low'
}

class DetectorError(ValueError):
    pass

class PromptManager:
    @staticmethod
    def system_prompt():
        return Path(__file__).resolve().with_name('prompt.txt').read_text(encoding='utf-8-sig').strip()

    @staticmethod
    def user_prompt(transcript):
        return ('Analyze the following transcript using the system instructions. '
                'Return exactly the 14 required JSON keys. Evidence values must be '
                'exact substrings of this transcript; summary must be exactly two sentences.\n'
                '<transcript>\n'+transcript+'\n</transcript>')

class JsonParser:
    @staticmethod
    def parse(response):
        if not isinstance(response,str):
            raise DetectorError('LLM response is not text')
        response = re.sub(r'<think\b[^>]*>.*?</think\s*>','',response,flags=re.S|re.I)
        decoder = json.JSONDecoder()
        for match in re.finditer(r'\{',response):
            try:
                obj,_ = decoder.raw_decode(response[match.start():])
                if isinstance(obj,dict):
                    return obj
            except json.JSONDecodeError:
                pass
        raise DetectorError('No valid JSON object in model response')

class Validator:
    @staticmethod
    def validate(raw:Dict[str,Any], transcript:str):
        if not isinstance(raw,dict):
            raise DetectorError('Model output must be a JSON object')
        missing = set(DEFAULTS)-set(raw)
        if missing:
            raise DetectorError('Missing required keys: '+', '.join(sorted(missing)))
        result={k:raw[k] for k in DEFAULTS}
        for key,allowed in ENUMS.items():
            if not isinstance(result[key],str) or result[key] not in allowed:
                raise DetectorError(f'Invalid {key}: {result[key]!r}')
        for key in ('agent_offered_booking','revenue_opportunity'):
            if type(result[key]) is not bool:
                raise DetectorError(f'{key} must be boolean')
        if type(result['intent_strength']) is not int or not 1<=result['intent_strength']<=5:
            raise DetectorError('intent_strength must be integer 1-5')
        if result['refusal_paraphrase'] is not None and not isinstance(result['refusal_paraphrase'],str):
            raise DetectorError('refusal_paraphrase must be text or null')
        evidence=result['evidence']
        if not isinstance(evidence,dict):
            raise DetectorError('evidence must be a field-to-quote object')
        repaired={}
        for field,quote in evidence.items():
            if field not in DEFAULTS or field=='evidence' or not isinstance(quote,str) or not quote.strip():
                continue
            quote=quote.strip()
            if quote in transcript:
                repaired[field]=quote
                continue
            # Recover whitespace-only formatting changes while preserving exact original substring.
            pattern=r'\s+'.join(re.escape(part) for part in quote.split())
            match=re.search(pattern,transcript)
            if match:
                repaired[field]=match.group(0)
            else:
                logger.warning('Dropping non-verbatim evidence for field %s',field)
        result['evidence']=repaired
        summary=result['summary']
        if not isinstance(summary,str) or not summary.strip():
            raise DetectorError('Missing summary')
        sentences=re.split(r'(?<=[.!?])\s+',summary.strip())
        if len(sentences)!=2 or not all(s.strip().endswith(('.', '!', '?')) for s in sentences):
            raise DetectorError('Summary must contain exactly two sentences')
        return result

class RevenueOpportunityAnalyzer:
    def __init__(self,llm,max_attempts=2):
        self.llm=llm
        self.max_attempts=max(1,int(max_attempts))

    def analyze(self,transcript:str)->Dict[str,Any]:
        transcript='' if transcript is None else str(transcript).strip()
        if not transcript or transcript.lower() in {'nan','null','none'}:
            raise DetectorError('Empty transcript')
        system=PromptManager.system_prompt()
        user=PromptManager.user_prompt(transcript)
        last_error=None
        for attempt in range(self.max_attempts):
            try:
                response=self.llm.generate(system_prompt=system,user_prompt=user,temperature=0.0,max_tokens=2000)
                return Validator.validate(JsonParser.parse(response),transcript)
            except (DetectorError,ValueError,TypeError) as exc:
                last_error=exc
                logger.warning('Detector attempt %s/%s failed: %s',attempt+1,self.max_attempts,exc)
                user=PromptManager.user_prompt(transcript)+('\nPrevious response failed validation: '
                      +str(exc)+'\nRegenerate a complete, valid JSON object. Copy evidence quotes verbatim.')
        raise DetectorError(f'Classification failed after {self.max_attempts} attempts: {last_error}')

    __call__=analyze
