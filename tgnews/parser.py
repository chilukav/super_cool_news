import enum
import string


class State(enum.IntEnum):
    TEXT = 0
    START_TAG_1 = 1
    START_TAG_2 = 1
    TAG_NAME = 2
    END_TAG_NAME_1 = 3
    END_TAG_NAME_2 = 3


START_TAG_ELEMENT_1 = b'</'
START_TAG_ELEMENT_2 = b'<'
START_TAG_ELEMENT = {
    State.START_TAG_1: START_TAG_ELEMENT_1,
    State.START_TAG_2: START_TAG_ELEMENT_2
}

END_TAG_ELEMENT_1 = b'>'
END_TAG_ELEMENT_2 = b'/>'

START_TAG_ELEMENT_1_SECOND = b'/'
END_TAG_ELEMENT_2_SECOND = b'>'

ASCII_LETTERS = set(string.ascii_letters.encode('ascii'))


def parse(text: bytes) -> bytes:
    state = State.TEXT
    tag_name = []
    text_without_tags = []

    for idx in range(0, len(text)):
        if state == State.TEXT:
            if text.startswith(START_TAG_ELEMENT_1, idx):
                state = State.START_TAG_1
                tag_name.extend(START_TAG_ELEMENT_1)
            elif text.startswith(START_TAG_ELEMENT_2, idx):
                state = State.START_TAG_2
                tag_name.extend(START_TAG_ELEMENT_2)
            else:
                text_without_tags.append(text[idx])

        elif state == State.START_TAG_1 or state == State.START_TAG_2:
            if state == State.START_TAG_1 and text.startswith(START_TAG_ELEMENT_1_SECOND, idx):
                continue
            if text[idx] in ASCII_LETTERS:
                state = State.TAG_NAME
                tag_name.append(text[idx])
            else:
                text_without_tags.append(text[idx])
                state = State.TEXT
                tag_name.clear()

        elif state == State.TAG_NAME:
            if text.startswith(END_TAG_ELEMENT_1, idx):
                state = State.TEXT
                tag_name.clear()
            elif text.startswith(END_TAG_ELEMENT_2, idx):
                if text[idx] == END_TAG_ELEMENT_2_SECOND:
                    continue
                state = State.TEXT
                tag_name.clear()
            elif text.startswith(START_TAG_ELEMENT_1, idx):
                state = State.START_TAG_1
                text_without_tags.extend(tag_name)
                tag_name.clear()
                tag_name.extend(START_TAG_ELEMENT_1)
            elif text.startswith(START_TAG_ELEMENT_2, idx):
                state = State.START_TAG_2
                text_without_tags.extend(tag_name)
                tag_name.clear()
                tag_name.extend(START_TAG_ELEMENT_2)
            else:
                tag_name.append(text[idx])

    text_without_tags.extend(tag_name)
    return bytes(text_without_tags)
