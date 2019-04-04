from pathlib import PosixPath, Path


class Errors:
    LINE = "="*25
    BASE_MSG = "\n{line}\n".format(line=LINE)

    def __call__(self, msg, exception):
        return exception(msg)

    def FileNotFound(self, path: PosixPath):
        path = Path(path)
        msg = self.BASE_MSG
        msg += "NOT Exists Path:\n"

        path_gradually = Path(path.parts[0])
        for path_part in path.parts[1:]:
            path_gradually /= path_part
            msg += "\tExists: {}, {}\n".format(path_gradually.exists(), path_gradually)

        return FileNotFoundError(msg)