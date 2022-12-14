import pandas as pd
import re


class TaskOrientedChatbot:
    def __init__(self,  path: str="."):
        self._done = True
        self._collected_data = self._init_data()
        self._stage = 0
        "possible stages:\n"
        "        0. ask to confirm rule_based start\n"
        "        1. confirm start\n"
        "        2. ask for information\n"
        "           provide answer\n"
        "        3. confirm end\n"
        "           ask for rechecking\n"
        "        4. recheck information\n"
        self._current_field = None
        self.schedule = pd.merge(pd.merge(pd.read_csv(f"{path}/data/rule_based/schedule.csv", header=0),
                                          pd.read_csv(f"{path}/data/rule_based/course.csv", header=0)),
                                 pd.read_csv(f"{path}/data/rule_based/tutors.csv", header=0))

    def is_completed(self):
        return self._done

    @staticmethod
    def _init_data():
        return dict(
            course=dict(
                data_type=int,
                value=None,
                question="O jaki przedmiot pytasz?",
                check=False
            ),
            tutor=dict(
                data_type=int,
                value=None,
                question="Kto prowadzi zajęcia?",
                check=False
            ),
            day=dict(
                data_type=int,
                value=None,
                question="Którego dnia odbywają się zającia?",
                check=False
            ),
            start_hour=dict(
                data_type=int,
                value=None,
                question="O której rozpoczynają się zającia?",
                check=False
            ),
            group=dict(
                data_type=int,
                value=None,
                question="O którą grupę pytasz?",
                check=False
            )
        )

    def _detect_confirmation(self, prompt: str) -> bool | type(None):
        result = None
        if 'tak' in prompt.lower():
            result = True
        elif 'nie' in prompt.lower():
            result = False
        return result

    def _ask_for_data(self):
        for k, v in self._collected_data.items():
            if v["value"] is None:
                self._current_field = k
                return v['question']

    def _retrive_info(self, prompt: str) -> type(None):
        # needs to be a bit more readable but works
        if self._current_field is not None:
            for v in self.schedule[self._current_field].values:
                # TODO look for aliases
                aliases = [c for c in self.schedule.columns if re.compile(self._current_field).match(c)]
                for v2 in self.schedule.loc[self.schedule[self._current_field] == v, aliases].values.reshape(-1):
                    if str(v2).lower() in prompt.lower():
                        self._collected_data[self._current_field]['value'] = v
        return None

    def _check_answer(self, recheck: bool = False) -> int | type(None):
        # needs to be a bit more readable but works
        res = 0
        curr_view = self.schedule.copy()
        for k, v in self._collected_data.items():
            if (v['value'] is None) | (recheck and (not v['check'])):
                res = None
            else:
                curr_view = curr_view.loc[curr_view[k] == v['value'], :]
        if curr_view.shape[0] == 1:
            res = curr_view['room'].values[0]
        return res

    def _check_data(self) -> type(None):
        for k, v in self._collected_data.items():
            if not v["check"]:
                self._current_field = k
                return f"Czy {k} to {v['value']}"  # TODO: provide polish names
        return None

    def _reset(self) -> str:
        self._done = True
        self._collected_data = self._init_data()
        self._stage = 0
        return "To o czym my tu rozmawialiśmy?"

    def interact(self, prompt: str) -> str:
        response = ""
        if self._stage == 0:  # ask to confirm rule_based start
            self._stage = 1
            self._done = False
            response = "Czy chcesz żebym sprawdził, gdzie masz zajęcia?"

        elif self._stage == 1:  # confirm start
            confirmation = self._detect_confirmation(prompt)
            if confirmation is None:
                response = "Nie rozumiem. Czy chcesz żebym sprawdził, gdzie masz zajęcia?"
            elif confirmation:
                self._stage = 2
            else:
                response = self._reset()

        if self._stage == 2:  # ask for information
            # save data from prompt
            self._retrive_info(prompt)

            # check if you can answer
            ans = self._check_answer()
            if ans is None:
                response = self._ask_for_data()
            else:
                response = f"Zajęcia odbywają się w sali {ans}. Czy to było pomocne?"
                self._stage = 3

        elif self._stage == 3:  # confirm end
            confirmation = self._detect_confirmation(prompt)
            if confirmation is None:
                response = "Nie rozumiem. Czy odpowiedź jest pomocna?"
            elif confirmation:
                response = self._reset()
            else:
                response = "Czy chcesz sprawdzić jeszcze raz?"
                self._stage = 4

        elif self._stage == 4:  # recheck info
            confirmation = self._detect_confirmation(prompt)
            if confirmation:
                self._collected_data = self._init_data()
                self._stage = 2
                response = self._ask_for_data()
            else:
                response = self._reset()

        return response
