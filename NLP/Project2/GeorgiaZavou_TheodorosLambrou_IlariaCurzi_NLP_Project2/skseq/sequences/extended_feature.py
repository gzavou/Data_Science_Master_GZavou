from skseq.sequences.id_feature import IDFeatures
import string

# === Custom linguistic lists ===
PREPOSITIONS = {"in", "on", "at", "from", "to", "with", "by", "of", "for", "about"}
NAME_TITLES = {"Mr.", "Mrs.", "Ms.", "Dr.", "Prof."}
MONTHS = {"January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"}
DAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}

# === Light affix lists ===
SUFFIXES = ["son", "ville", "land", "ford", "burg", "stein", "man", "corp", "inc"]
PREFIXES = ["Mc", "O'", "Van", "San", "St", "Dr"]

class ExtendedFeatures(IDFeatures):
    """
    Custom feature extractor for structured perceptron.
    Adds lexical, morphological, affixal and contextual features for improved NER generalization.
    """

    def add_emission_features(self, sequence, pos, y, features):
        x = sequence.x[pos]
        y_name = self.dataset.y_dict.get_label_name(y)

        word = x if isinstance(x, str) else self.dataset.x_dict.get_label_name(x)
        word = str(word)

        # --- Basic identity ---
        feat_id = self.add_feature(f"id:{word}::{y_name}")
        if feat_id != -1:
            features.append(feat_id)

        # --- Hyphen ---
        if "-" in word:
            features.append(self.add_feature(f"hyphen::{y_name}"))

        # --- Morphological / lexical ---
        if word.istitle():
            features.append(self.add_feature(f"title_case::{y_name}"))
        if word.isupper():
            features.append(self.add_feature(f"all_caps::{y_name}"))
        if word.islower():
            features.append(self.add_feature(f"all_lower::{y_name}"))
        if word.isdigit():
            features.append(self.add_feature(f"digit_only::{y_name}"))
        if any(c.isdigit() for c in word) and any(c.isalpha() for c in word):
            features.append(self.add_feature(f"alnum_mix::{y_name}"))
        if len(word) <= 3:
            features.append(self.add_feature(f"short_word::{y_name}"))
        elif len(word) <= 6:
            features.append(self.add_feature(f"medium_word::{y_name}"))
        else:
            features.append(self.add_feature(f"long_word::{y_name}"))
        if any(c.isupper() for c in word[1:]):
            features.append(self.add_feature(f"internal_cap::{y_name}"))
        if any(word.count(c) > 2 for c in set(word)):
            features.append(self.add_feature(f"char_repeat::{y_name}"))
        shape = self.get_word_shape(word)
        features.append(self.add_feature(f"shape={shape}::{y_name}"))

        # --- Sentence position ---
        if pos == 0:
            features.append(self.add_feature(f"start_of_sentence::{y_name}"))
        elif pos == len(sequence.x) - 1:
            features.append(self.add_feature(f"end_of_sentence::{y_name}"))
        if pos > 0 and word.istitle():
            features.append(self.add_feature(f"mid_sentence_cap::{y_name}"))

        # --- Contextual features (refined) ---
        if pos > 0:
            prev = sequence.x[pos - 1]
            prev_word = prev if isinstance(prev, str) else self.dataset.x_dict.get_label_name(prev)
            prev_word = str(prev_word)

            if prev_word.istitle():
                features.append(self.add_feature(f"prev_title_case::{y_name}"))
            if prev_word.isupper():
                features.append(self.add_feature(f"prev_all_caps::{y_name}"))
            if prev_word.lower() in NAME_TITLES:
                features.append(self.add_feature(f"prev_is_title_trigger::{y_name}"))
            if prev_word.lower() in PREPOSITIONS:
                features.append(self.add_feature(f"prev_is_preposition::{y_name}"))

        if pos < len(sequence.x) - 1:
            next_ = sequence.x[pos + 1]
            next_word = next_ if isinstance(next_, str) else self.dataset.x_dict.get_label_name(next_)
            next_word = str(next_word)

            if next_word.istitle():
                features.append(self.add_feature(f"next_title_case::{y_name}"))
            if next_word.isupper():
                features.append(self.add_feature(f"next_all_caps::{y_name}"))
            if next_word.lower() in {"inc", "ltd", "corporation", "company"}:
                features.append(self.add_feature(f"next_is_org_suffix::{y_name}"))
            if next_word.lower() in MONTHS:
                features.append(self.add_feature(f"next_is_month::{y_name}"))


        # --- Linguistic semantic features ---
        if word in PREPOSITIONS:
            features.append(self.add_feature(f"is_preposition::{y_name}"))
        if word in NAME_TITLES:
            features.append(self.add_feature(f"is_name_title::{y_name}"))
        if word in MONTHS:
            features.append(self.add_feature(f"is_month::{y_name}"))
        if word in DAYS:
            features.append(self.add_feature(f"is_weekday::{y_name}"))

        # --- Affix-based features ---
        for suff in SUFFIXES:
            if word.lower().endswith(suff.lower()):
                features.append(self.add_feature(f"suffix_{suff.lower()}::{y_name}"))
                break  # avoid multiple suffix hits

        for pref in PREFIXES:
            if word.startswith(pref):
                features.append(self.add_feature(f"prefix_{pref.lower()}::{y_name}"))
                break

        return features

    def get_word_shape(self, word):
        """
        Returns a simplified shape of the word:
        - Uppercase -> 'X', lowercase -> 'x', digit -> 'd', punctuation -> '.', other -> '#'
        """
        shape = ""
        for char in word:
            if char.isupper():
                shape += "X"
            elif char.islower():
                shape += "x"
            elif char.isdigit():
                shape += "d"
            elif char in string.punctuation:
                shape += "."
            else:
                shape += "#"
        return shape[:6]
