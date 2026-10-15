CSR: Thank you for calling Service Experts. This is Aaron. CSR: How can I help you? CUSTOMER: Yes, I'd like to see how we can change ownership to another owner, CUSTOMER: another account. CSR: Oh, okay. CSR: Well, CSR: let me go ahead and pull up your account first here on my end. CSR: And right now, can you provide me your service address? CUSTOMER: It's [SERVICE_ADDRESS] CUSTOMER: Texas zip code CSR: Okay. CUSTOMER: [ZIP_CODE]. CSR: All right. CSR: Thank you so much for that. CSR: And how about your first and last name? CUSTOMER: My name is [PERSON_NAME]. CSR: Thank you for that. And can you also confirm to me the name that is set up on the account? CUSTOMER: Senator Anita Pena? CSR: Yes, thank you so much for that, Jesus. CSR: And thank you as well for being an advantage member with us. CSR: So you've mentioned earlier that you're inquiring about transferring the ownership of the account to another owner, CSR: right? CUSTOMER: Yes, we did that. We did that in June, and they're ready to transfer to... CUSTOMER: Anthony X home he CSR: Hmm. CSR: Okay, CSR: Snap apps like a new buyer or like a new house or home owner. CUSTOMER: took over the whole thing he took over the bill he took over the air conditioner he took he took over everything CSR: Oh, okay. CSR: Well, right now since you're under the advantage plan or the monthly payments for the equipment, CSR: there should be or it would be like another or a different process from that. CSR: But here on our end, we'll be just changing the name over on that one. CSR: And right now that may connect you over to the advantage team so they can provide you like the process on transferring over the ownership. CSR: And with that, I just want to just connect you over to them. CSR: Just want to confirm the contact details we have on file, CSR: which we have a phone number. CSR: It's [PHONE_NUMBER]. Is this the best number? CUSTOMER: Yes, sir. CUSTOMER: Yes, sir. CSR: Thank you. CSR: And email address is unicorn or 490 at gmail.com. Active email. CUSTOMER: Yes. CUSTOMER: Yes, CSR: All CUSTOMER: all CSR: right. CSR: So with that, this is this day on the line. Let me now connect you over to the Advantage team. CSR: Just a moment. CUSTOMER: right. CUSTOMER: Morning. CUSTOMER: Yes. CUSTOMER: Yes, you are. CUSTOMER: Can you hear me? CUSTOMER: Excuse me? CUSTOMER: Yeah, CUSTOMER: I can hear you. CUSTOMER: Yes, yes, hang on. CSR: Mm-hmm. CUSTOMER: If I could, okay, CUSTOMER: can you hear me now? CSR: Okay. CSR: I guess actually I'm having a hard time to hear your property, CSR: but can you hear me? CUSTOMER: Yes, I can hear you. CUSTOMER: Can you hear me now? CSR: Yes, CSR: yes. CSR: Now I can hear you. CUSTOMER: Okay, yeah, CUSTOMER: we sold the house in June. CUSTOMER: Yeah, around June. CSR: Mm CUSTOMER: We CSR: -hmm. CUSTOMER: assume that everything got transferred already and we've been getting costs or the payments not being paid. CUSTOMER: And I thought we agreed to transfer ownership to Anthony X Homes. CUSTOMER: Do you guys have that information? CSR: Okay. CSR: Yes, upon checking here, yes, there is a transfer request here. CSR: Even before, that was last June. CSR: Yeah, CSR: there was a transfer request last June for the account. CSR: It was not properly transferred. It was not successfully transferred to the new buyer or to the new homeowner since in regards with the process of transfer, CSR: you must make sure first or we must make sure first that the account was current for us to be able to process that one. So that is the only thing that I think was not being settled because upon checking here, there's still a balance here in the account. CSR: That's why we cannot able to process a transfer of ownership. CSR: ownership. CUSTOMER: Well, CUSTOMER: he was making payments. He stopped making payments around September. CUSTOMER: And he made no payments. CUSTOMER: So he actually stopped making payments at September. CSR: By September, CUSTOMER: And then he there, CSR: said, you mean she starts... CUSTOMER: he was making payments, not us. He was making payments to the system. CUSTOMER: He was making payments. CSR: Okay, CSR: even if it's okay, CSR: let me check that one because even if it's up processing a payment for the. CSR: Oh, yeah, I saw with you that he processed a payment for the amount of $728.54. CSR: But right now, just like what I said, since the account was still behind. CSR: So that's why there's still a volunteer in the account that needs to be settled. CSR: That's why we cannot offer the transfer of ownership. CSR: It needs to settle first before we can able to process the transfer. CUSTOMER: Well, okay, CUSTOMER: I'm going to give him a call. CUSTOMER: And I'm going to make him pay everything that he owes you, CUSTOMER: and then we'll do a transfer, CSR: Yes, CUSTOMER: because CSR: correct. CUSTOMER: we're not, this is no longer our house. CUSTOMER: It's no longer our place CSR: Okay, CUSTOMER: no more. CUSTOMER: So he's responsible for the CSR: I understand. CUSTOMER: payment. CSR: Yes, correct. CUSTOMER: Okay. CSR: So I would like you to reach out for the new homeowner right now and inform them that there is a balance hitting the account. CSR: And if they want the property transferred under their name, they must settle it first. CSR: Okay? CSR: And you can just give us a call back again. CUSTOMER: Do you have this number? CSR: Sorry, is there anything else? CUSTOMER: Do you have this number? CSR: Let me try to check. CSR: If you already provided that information, just give me a moment. CSR: transfer okay CSR: with a moment I has this okay I'm just trying to check if CUSTOMER: Okay. CSR: there is already the information called in transfer Anthony CSR: Anthony CUSTOMER: Yes. CSR: Your home owner is Anthony. CSR: Okay. CSR: And do you have the phone number here? CSR: Yeah, CSR: we have the phone number, CSR: we have the name, CSR: but do you have his last name by any chance? CUSTOMER: No, CUSTOMER: we don't have his last name. CUSTOMER: All I think is put X homes. CSR: Okay. CSR: So I would like you to please reach out to them. CSR: I'm his sister. CSR: You can reach out to Anthony and inform him that we need first to settle the volunteering the account if you really want to properly transfer the ownership under his name. CSR: When we already make the account back to current, CSR: then that's the time you can able to process a transfer and we can send to you a transfer letter to you and to Anthony as well so you can sign it and send it back to us so we can properly transfer it under his name already. CSR: Okay? CUSTOMER: Okay, okay. CUSTOMER: Can you go give them a call or okay? CSR: Actually, I ask you to, if it's okay if you're going to give him a call and inform him regarding about this one, CSR: okay? CUSTOMER: Okay. All right. CSR: Thank you. CSR: So is there anything else as us? CUSTOMER: No, that's it. That's it. CUSTOMER: Thank you. CSR: You're welcome. CSR: It's been a pleasure speaking with you and have a great day. Bye-bye. CUSTOMER: Bye.


"""
==============================================================================
Track A - Service Experts Revenue Opportunity Detection

LLM Client

Responsibilities
----------------
1. Build chat messages
2. Call ModelRegistry
3. Normalize Databricks Foundation Model responses
4. Return clean text

Business logic is NOT handled here.

Author
------
Ravindra Gantasala
==============================================================================
"""

import logging
import re
from typing import Dict, List, Optional

from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Centralized client for all LLM inference.

    Every downstream module should call ONLY this class.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        endpoint_name: Optional[str] = None
    ):

        self.registry = registry
        self.endpoint_name = endpoint_name

        logger.info("LLM Client initialized.")

    # ------------------------------------------------------------------
    # Build Messages
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> List[Dict]:

        return [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ]

    # ------------------------------------------------------------------
    # Public Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> str:

        messages = self._build_messages(
            system_prompt,
            user_prompt
        )

        raw_response = self.registry.predict_llm(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            endpoint_name=self.endpoint_name
        )

        return self._extract_text(raw_response)

    # ------------------------------------------------------------------
    # Databricks Response Parser
    # ------------------------------------------------------------------

    def _extract_text(self, response) -> str:
        """
        Normalizes every Databricks response shape.

        Supported

        Chat Endpoint

        {
            "choices":[
                {
                    "message":{
                        "content":"..."
                    }
                }
            ]
        }

        Foundation Models

        {
            "predictions":[
                {
                    "content":"..."
                }
            ]
        }

        {
            "predictions":[
                {
                    "text":"..."
                }
            ]
        }

        Legacy

        {
            "predictions":[ "..."]
        }
        """

        logger.info("=" * 80)
        logger.info("RAW LLM RESPONSE")
        logger.info(response)
        logger.info("=" * 80)

        text = None

        try:

            # ----------------------------------------------------------
            # OpenAI Chat schema
            # ----------------------------------------------------------

            if isinstance(response, dict):

                choices = response.get("choices")

                if choices:

                    message = choices[0].get("message", {})

                    text = message.get("content")

                # ------------------------------------------------------
                # Databricks Foundation Model schema
                # ------------------------------------------------------

                if text is None:

                    predictions = response.get("predictions")

                    if predictions:

                        first = predictions[0]

                        if isinstance(first, dict):

                            text = (
                                first.get("content")
                                or first.get("text")
                                or first.get("generated_text")
                            )

                        elif isinstance(first, str):

                            text = first

                # ------------------------------------------------------
                # Direct Content
                # ------------------------------------------------------

                if text is None:

                    text = (
                        response.get("content")
                        or response.get("text")
                    )

            elif isinstance(response, str):

                text = response

            else:

                text = str(response)

        except Exception:

            logger.exception("Unable to parse model response.")

            text = str(response)

        if text is None:
            text = ""

        return self._clean_response(text)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _clean_response(
        self,
        text: str
    ) -> str:

        if not text:
            return ""

        if isinstance(text, list):
            text = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            )

        text = text.strip()

        text = re.sub(
            r"^```json",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"^```",
            "",
            text
        )

        text = re.sub(
            r"```$",
            "",
            text
        )

        text = text.replace("<think>", "")
        text = text.replace("</think>", "")

        return text.strip()




##########################
"""
==============================================================================
Track A - Service Experts Revenue Opportunity Detection

LLM Client

Responsibilities
----------------
1. Build chat messages
2. Call ModelRegistry
3. Normalize Databricks Foundation Model responses
4. Return clean text

Business logic is NOT handled here.

Author
------
Ravindra Gantasala
==============================================================================
"""

import logging
import re
from typing import Dict, List, Optional

from models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Centralized client for all LLM inference.

    Every downstream module should call ONLY this class.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        endpoint_name: Optional[str] = None
    ):

        self.registry = registry
        self.endpoint_name = endpoint_name

        logger.info("LLM Client initialized.")

    # ------------------------------------------------------------------
    # Build Messages
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> List[Dict]:

        return [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ]

    # ------------------------------------------------------------------
    # Public Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> str:

        messages = self._build_messages(
            system_prompt,
            user_prompt
        )

        raw_response = self.registry.predict_llm(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            endpoint_name=self.endpoint_name
        )

        return self._extract_text(raw_response)

    # ------------------------------------------------------------------
    # Databricks Response Parser
    # ------------------------------------------------------------------

    def _extract_text(self, response) -> str:
        """
        Normalizes every Databricks response shape.

        Supported

        Chat Endpoint

        {
            "choices":[
                {
                    "message":{
                        "content":"..."
                    }
                }
            ]
        }

        Foundation Models

        {
            "predictions":[
                {
                    "content":"..."
                }
            ]
        }

        {
            "predictions":[
                {
                    "text":"..."
                }
            ]
        }

        Legacy

        {
            "predictions":[ "..."]
        }
        """

        # The model output may repeat PII from the call. Never log it verbatim.
        logger.debug("LLM response received, type=%s", type(response).__name__)

        text = None

        try:

            # ----------------------------------------------------------
            # OpenAI Chat schema
            # ----------------------------------------------------------

            if isinstance(response, dict):

                choices = response.get("choices")

                if choices:

                    message = choices[0].get("message", {})

                    text = message.get("content")

                # ------------------------------------------------------
                # Databricks Foundation Model schema
                # ------------------------------------------------------

                if text is None:

                    predictions = response.get("predictions")

                    if predictions:

                        first = predictions[0]

                        if isinstance(first, dict):

                            text = (
                                first.get("content")
                                or first.get("text")
                                or first.get("generated_text")
                            )

                        elif isinstance(first, str):

                            text = first

                # ------------------------------------------------------
                # Direct Content
                # ------------------------------------------------------

                if text is None:

                    text = (
                        response.get("content")
                        or response.get("text")
                    )

            elif isinstance(response, str):

                text = response

            else:

                text = str(response)

        except Exception:

            logger.exception("Unable to parse model response.")

            text = str(response)

        if text is None:
            text = ""

        return self._clean_response(text)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _clean_response(
        self,
        text: str
    ) -> str:

        if not text:
            return ""

        if isinstance(text, list):
            text = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            )

        text = text.strip()

        text = re.sub(
            r"^```json",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"^```",
            "",
            text
        )

        text = re.sub(
            r"```$",
            "",
            text
        )

        # Remove complete reasoning/thinking blocks, not only their markers.
        # A leftover reasoning narrative would otherwise make JSON parsing fail.
        text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text,
                      flags=re.DOTALL | re.IGNORECASE)

        return text.strip()
