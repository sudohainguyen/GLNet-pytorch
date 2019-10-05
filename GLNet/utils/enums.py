import enum

class PhaseMode(enum.Enum):
    GlobalOnly = 1
    LocalFromGlobal = 2
    GlobalFromLocal = 3

    @staticmethod
    def int_to_phasemode(mode):
        if mode == 1:
            return PhaseMode.GlobalOnly
        if mode == 2:
            return PhaseMode.LocalFromGlobal
        return PhaseMode.GlobalFromLocal
