from enum import Enum


class AttackResult(Enum):
    SUCCESS = 0
    SKIPPED = 1
    FAILED = 2
    ERROR = 3

    @classmethod
    def from_mapping_string(cls, map_string, sample):
        """
            Computes the attack result from a string in the formats

            LABEL, ..., LABEL --> LABEL, ..., LABEL 
            LABEL, ..., LABEL --> [SKIPPED/FAILED]
        """
        if "Could not attack sample:" in sample:
            return AttackResult.FAILED

        _, dest = map_string.split(" --> ")

        if "FAILED" in dest:
            return AttackResult.FAILED
        elif "SKIPPED" in dest:
            return AttackResult.SKIPPED
        else:
            return AttackResult.SUCCESS

    @classmethod
    def from_textattack_class(cls, textattack_cls):
        """
            Computes the attack result from a string in the formats

            LABEL, ..., LABEL --> LABEL, ..., LABEL 
            LABEL, ..., LABEL --> [SKIPPED/FAILED]
        """
        if textattack_cls == "ErrorAttackResult":
            return AttackResult.ERROR
        elif textattack_cls == "SkippedAttackResult":
            return AttackResult.SKIPPED
        elif "FailedAttackResult" in textattack_cls:
            return AttackResult.FAILED
        elif "SuccessfulAttackResult" in textattack_cls:
            return AttackResult.SUCCESS

        raise Exception("The provided class is not mapped")
