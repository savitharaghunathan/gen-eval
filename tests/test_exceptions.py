from geneval.exceptions import ProfileNotFoundError, ProfileValidationError, UnknownMetricError


class TestProfileValidationError:
    def test_is_value_error(self):
        err = ProfileValidationError("weights don't sum to 1.0")
        assert isinstance(err, ValueError)

    def test_message(self):
        err = ProfileValidationError("weights for 'rigorous_rag' sum to 0.8, expected 1.0")
        assert "rigorous_rag" in str(err)
        assert "0.8" in str(err)


class TestUnknownMetricError:
    def test_is_value_error(self):
        err = UnknownMetricError("hallucination_score")
        assert isinstance(err, ValueError)

    def test_message(self):
        err = UnknownMetricError("hallucination_score", available=["faithfulness", "answer_relevancy"])
        assert "hallucination_score" in str(err)
        assert "faithfulness" in str(err)


class TestProfileNotFoundError:
    def test_is_key_error(self):
        err = ProfileNotFoundError("nonexistent_profile")
        assert isinstance(err, KeyError)

    def test_message(self):
        err = ProfileNotFoundError("nonexistent_profile")
        assert "nonexistent_profile" in str(err)
