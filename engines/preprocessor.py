from inscriptis import get_text
import pathlib


class PreProcessor:
    def __init__(self, document):
        self.document = document

    def process(self):
        text = get_text(pathlib.Path(self.document).read_text())
        return text
