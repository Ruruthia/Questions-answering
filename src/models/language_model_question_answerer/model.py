
class LanguageModelQuestionAnswerer:

    def __init__(self, language_model, clusterer, num_questions_to_sample, generation_config):
        self._language_model = language_model
        self._clusterer = clusterer
        self._num_questions_to_sample = num_questions_to_sample
        self._generation_config = generation_config

    def _prepare_prompt(self, question: str):
        """
        Gets questions similar to the question
        and prepares the prompt by processing the questions & answers and joining them using the EOS token.
        """
        cluster_id = self._clusterer.cluster_single_question(question)
        sampled_qas = self._clusterer.sample_questions_from_cluster(cluster_id, self._num_questions_to_sample)
        return " ".join(
            [f"{qa.question} Odpowiedź: {qa.answers[0]}" for qa in sampled_qas]
            + [f"{question} Odpowiedź: "]
        )

    def answer_question(self, question):
        prompt = self._prepare_prompt(question)
        response = self._language_model.respond_to_prompt(
            prompt=prompt,
            generation_config=self._generation_config,
            end_sequence="###",
        )[0]
        # Get first word of the response
        response = response[len(prompt):].split(' ')[0]
        return response
