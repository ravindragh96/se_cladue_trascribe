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
