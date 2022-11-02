class TaskOrientedChatbot:
    def __init__(self):
        self._done = False
        self._collected_data = self._init_data()
        self._stage = 0
        "possible stages:\n"
        "        0. ask to confirm rule_based start\n"
        "        1. confirm start\n"
        "        2. ask for information\n"
        "           provide answer\n"
        "        3. confirm end\n"
        "           ask for rechecking\n"
        "        4. confirm rechecking\n"
        "        5. recheck information\n"
        self._current_field = None


    def is_completed(self):
        return self._done

    @staticmethod
    def _init_data():
        return dict(
            group=dict(
                data_type=int,
                value=None,
                question="O którą grupę pytasz?",
                priority=1,
                check=False
            ),
            tutor=dict(
                data_type=int,
                value=None,
                question="Kto prowadzi zajęcia?",
                priority=1,
                check=False
            ),
            course=dict(
                data_type=int,
                value=None,
                question="O jaki przedmiot pytasz?",
                priority=1,
                check=False
            ),
            room=dict(
                data_type=int,
                value=None,
                question="W jakiej sali odbywają się zajęcia?",
                priority=1,
                check=False
            ),
            start_hour=dict(
                data_type=int,
                value=None,
                question="O której rozpoczynają się zającia?",
                priority=1,
                check=False
            )
        )

    def _detect_confirmation(self, prompt: str) -> bool | type(None):
        if 'tak' in prompt.lower():
            return True
        elif 'nie' in prompt.lower():
            return False
        return None

    def _ask_for_data(self):
        for k, v in self.collected_data.items():
            if v["value"] is None:
                self.current_field = k
                return v['question']

    def _retrive_info(self, prompt: str) -> type(None):
        return None


    def _check_answer(self, recheck: bool = False) -> bool | type(None):
        return True

    def _check_data(self) -> type(None):
        for k, v in self.collected_data.items():
            if not v["check"]:
                self.current_field = k
                return f"Czy {k} to {v['value']}"  # TODO: provide polish names
        return None

    def _reset(self) -> str:
        self.done = False
        self.collected_data = self._init_data()
        return "To o czym my tu rozmawialiśmy?"

    def interact(self, prompt: str) -> str:
        response = ""
        if self.stage == 0:  # ask to confirm rule_based start
            self.stage = 1
            response = "Czy chcesz żebym sprawdził, gdzie masz zajęcia?"

        elif self.stage == 1:  # confirm start
            confirmation = self._detect_confirmation(prompt)
            if confirmation is None:
                response = "Nie rozumiem. Czy chcesz żebym sprawdził, gdzie masz zajęcia?"
            elif confirmation:
                self.stage = 2
            else:
                response = self._reset()

        if self.stage == 2:  # ask for information
            # save data from prompt
            self._retrive_info(prompt)

            # check if you can answer
            ans = self._check_answer()
            if ans is None:
                response = self._ask_for_data()
            else:
                response = f"Zajęcia odbywają się w sali {ans}. Czy to było pomocne?"
                self.stage = 3

        elif self.stage == 3:  # confirm end
            confirmation = self._detect_confirmation(prompt)
            if confirmation is None:
                response = "Nie rozumiem. Czy odpowiedź jest pomocna?"
            elif confirmation:
                response = self._reset()
            else:
                response = "Czy chcesz sprawdzić jeszcze raz?"

        elif self.stage == 4:  # confirm rechecking
            confirmation = self._detect_confirmation(prompt)
            if confirmation:
                self.stage = 5
            else:
                response = self._reset()

        if self.stage == 5:  # recheck information
            self._retrive_info(prompt)

            ans = self._check_answer(recheck=True)
            if ans is None:
                response = self._check_data()
            else:
                response = f"Zajęcia odbywają się w sali {ans}. Czy to było pomocne?"
                self.stage = 3

        return response

