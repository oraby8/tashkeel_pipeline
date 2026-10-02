import re
import warnings
from pyarabic import araby
from prettytable import PrettyTable
from tqdm import tqdm

class ArabicDiacritizationEvaluator:
    STANDARD_ARABIC_LETTER_PATTERN = r"ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
    STANDARD_HARAKA_PATTERN = r'ًٌٍَُِّْٓ'
    TATWEEL_PATTERN = r"ـ"
    QURANIC_AND_ISLAMIC_ANNOTATION_SIGNS = r'ؘؙؚٰؐؑؒؓؕؗ٘ۖۗۘۙۚۛۜٱ۝۞ۣ۟۟۠ۡۢۤۥۦۧۨ۩۪ۭٕ۫۬۬ٗٔࣔࣕࣖࣗࣘࣙࣚࣛࣜࣝࣞࣟ࣠࣡࣢ࣰࣱࣲࣳ﴾﴿ﷰﷱﷲﷳﷳﷴﷵﷶﷷﷸﷺﷻ﷼﷽'
    NON_ARABIC_SYMOBOLS_IN_ARABIC_BLOCK = r"؀؁؂؃؄؅؎؏ؘؙؚؔؖ؋"
    NON_ARABIC_LETTERS_IN_ARABIC_UNICODE_BLOCK = r'ؠػؼؽؾؿٮٯٲٳٴٵٶٷٸٹٺٻټٽپٿڀځڂڃڄڅچڇڈډڊڋڌڍڎڏڐڑڒړڔڕږڗژڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨکڪګڬڭڮگڰڱڲڳڴڵڶڷڸڹںڻڼڽھڿۀہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓەۮۯۺۻۼ۽۾ۿ'
    NO_HARAKA = '*'
    ARABIC_LETTERS_PATTERN = r"[" + STANDARD_ARABIC_LETTER_PATTERN + "]+"

    FULLY_EXTENED_ARABIC_WORD = (
        STANDARD_ARABIC_LETTER_PATTERN +
        STANDARD_HARAKA_PATTERN +
        TATWEEL_PATTERN +
        QURANIC_AND_ISLAMIC_ANNOTATION_SIGNS +
        NON_ARABIC_SYMOBOLS_IN_ARABIC_BLOCK +
        NON_ARABIC_LETTERS_IN_ARABIC_UNICODE_BLOCK
    )

    @classmethod
    def split_arabic_text(cls, text: str) -> list:
        det_chars = cls.FULLY_EXTENED_ARABIC_WORD
        pattern = f"([{re.escape(det_chars)}]+)"
        result = re.split(pattern, text)
        return [word for word in result if word]

    @classmethod
    def extract_harakat(cls, word: str, shadda_is_letter: bool = False):
        harakat = []
        letters = []
        i = -1
        while word != "":
            char = word[0]
            word = word[1:]
            if araby.is_haraka(char) or araby.is_shadda(char):
                if len(letters) > 0:
                    if char == araby.SHADDA:
                        if shadda_is_letter:
                            if not letters[i].endswith(araby.SHADDA):
                                letters[i] = letters[i] + araby.SHADDA
                        else:
                            if harakat[i] == cls.NO_HARAKA:
                                harakat[i] = char
                    else:
                        if harakat[i] == cls.NO_HARAKA:
                            harakat[i] = char
                        elif harakat[i] == araby.SHADDA:
                            harakat[i] = harakat[i] + char
            else:
                letters.append(char)
                harakat.append(cls.NO_HARAKA)
                i += 1
        return harakat, letters

    @classmethod
    def has_arabic_letter(cls, s: str):
        result = re.findall(cls.ARABIC_LETTERS_PATTERN, s)
        return bool(result)

    @classmethod
    def has_al_alta3reef(cls, word: str, must_have_voweles: bool = False):
        harakat, letters = cls.extract_harakat(word)
        if len(letters) < 4:
            return False, -1
        no_harak_or_sukun = [cls.NO_HARAKA, araby.SUKUN]
        index = 0
        while True:
            if index >= len(letters): return False, -1
            if letters[index] == araby.WAW and harakat[index] in [cls.NO_HARAKA, araby.FATHA]: index += 1
            elif letters[index] == araby.FEH and harakat[index] in [cls.NO_HARAKA, araby.FATHA]: index += 1
            elif letters[index] == araby.ALEF_HAMZA_ABOVE and harakat[index] in [cls.NO_HARAKA, araby.FATHA]: index += 1
            else: break

        letters = letters[index:]
        harakat = harakat[index:]
        if len(letters) < 4: return False, -1
        next_letter_haraka = harakat[2]
        if letters[0] == araby.ALEF and letters[1] == araby.LAM:
            if harakat[0] == cls.NO_HARAKA:
                if letters[2] == araby.ALEF and harakat[1] == araby.KASRA: return True, index
                if must_have_voweles:
                    if (harakat[1] == cls.NO_HARAKA and araby.SHADDA in next_letter_haraka) or \
                       (harakat[1] == araby.SUKUN and araby.SHADDA not in next_letter_haraka):
                        return True, index
                else:
                    if harakat[1] in no_harak_or_sukun: return True, index
        return False, -1

    @classmethod
    def has_ll_alta3reef(cls, word: str, must_have_voweles: bool = False):
        harakat, letters = cls.extract_harakat(word)
        if len(letters) < 4: return False, -1
        no_harak_or_sukun = [cls.NO_HARAKA, araby.SUKUN]
        no_harak_or_fatha = [cls.NO_HARAKA, araby.FATHA]
        no_harak_or_kasra_or_fatha = [cls.NO_HARAKA, araby.KASRA, araby.FATHA]
        index = 0
        while True:
            if index >= len(letters): return False, -1
            if letters[index] == araby.WAW and harakat[index] in no_harak_or_fatha: index += 1
            elif letters[index] == araby.FEH and harakat[index] in no_harak_or_fatha: index += 1
            elif letters[index] == araby.ALEF_HAMZA_ABOVE and harakat[index] in no_harak_or_fatha: index += 1
            else: break
        letters = letters[index:]
        harakat = harakat[index:]
        if len(letters) < 4: return False, -1
        next_letter_haraka = harakat[2]
        if letters[0] == araby.LAM and letters[1] == araby.LAM:
            if harakat[0] in no_harak_or_kasra_or_fatha:
                if harakat[1] in no_harak_or_sukun or (letters[2] == araby.ALEF and harakat[1] == araby.KASRA):
                    return True, index
        return False, -1

    @classmethod
    def is_fully_diacritized(cls, word: str, count_last_haraka: bool = True) -> bool:
        harakat, letters = cls.extract_harakat(word)
        if len(letters) > 3:
            has_al, al_index = cls.has_al_alta3reef(word, must_have_voweles=True)
            if has_al and al_index + 2 <= len(harakat):
                harakat = harakat[:al_index] + harakat[al_index + 2:]
                letters = letters[:al_index] + letters[al_index + 2:]
        if len(letters) > 3:
            has_ll, ll_index = cls.has_ll_alta3reef(word, must_have_voweles=True)
            if has_ll and ll_index + 2 <= len(harakat):
                harakat = harakat[:ll_index] + harakat[ll_index + 2:]
                letters = letters[:ll_index] + letters[ll_index + 2:]
        if len(harakat) == 0: return False
        if not count_last_haraka:
            harakat = harakat[:-1]
            letters = letters[:-1]
        for i, h in enumerate(harakat):
            if h == cls.NO_HARAKA:
                prev_h = harakat[i - 1] if i > 0 else cls.NO_HARAKA
                if not (araby.is_alef(letters[i]) or (letters[i] == araby.WAW and araby.DAMMA in prev_h) or (letters[i] == araby.YEH and araby.KASRA in prev_h)):
                    return False
        return True

    @classmethod
    def caculate_error_on_single_sentence(cls, voweled_sentence: str, gt_sentence: str, gt_missing_diacritic_is_error: bool = False):
        voweled_words = cls.split_arabic_text(voweled_sentence)
        gt_words = cls.split_arabic_text(gt_sentence)
        if len(voweled_words) != len(gt_words):
            raise RuntimeError(f"words count does not match {len(voweled_words)} vs {len(gt_words)}")

        word_count, letter_count, not_voweled_words_count = 0, 0, 0
        total_wer, morph_wer, total_der, morph_der = 0, 0, 0, 0

        for i in range(len(voweled_words)):
            v_word = voweled_words[i].strip()
            gt_word = gt_words[i].strip()
            if not v_word and not gt_word: continue
            if not cls.has_arabic_letter(v_word): continue

            if not cls.is_fully_diacritized(v_word):
                not_voweled_words_count += 1

            v_harakat, v_letters = cls.extract_harakat(v_word)
            gt_harakat, gt_letters = cls.extract_harakat(gt_word)
            word_count += 1
            letter_count += len(gt_letters)

            if v_letters != gt_letters:
                raise RuntimeError(f"words don't match [{v_word}] , [{gt_word}]")

            correct_all_but_last = True
            correct_last = True

            if gt_missing_diacritic_is_error or cls.is_fully_diacritized(gt_word):
                for j in range(len(v_harakat) - 1):
                    if v_harakat[j] != gt_harakat[j]:
                        total_der += 1
                        morph_der += 1
                        correct_all_but_last = False
                if not correct_all_but_last:
                    morph_wer += 1
                if v_harakat[-1] != gt_harakat[-1]:
                    total_der += 1
                    correct_last = False
                if not (correct_last and correct_all_but_last):
                    total_wer += 1
            else:
                for j in range(len(v_harakat) - 1):
                    if v_harakat[j] != gt_harakat[j] and gt_harakat[j] == cls.NO_HARAKA:
                        letter_count -= 1
                        continue
                    if v_harakat[j] != gt_harakat[j]:
                        total_der += 1
                        morph_der += 1
                        correct_all_but_last = False
                if not correct_all_but_last:
                    morph_wer += 1
                if v_harakat[-1] != gt_harakat[-1] and gt_harakat[-1] == cls.NO_HARAKA:
                    letter_count -= 1
                else:
                    if v_harakat[-1] != gt_harakat[-1]:
                        total_der += 1
                        correct_last = False
                    if not (correct_last and correct_all_but_last):
                        total_wer += 1

        return word_count, letter_count, not_voweled_words_count, total_wer, morph_wer, total_der, morph_der

    @classmethod
    def caculate_errors_on_sentences(cls, voweled_sentences, ground_truth_sentences, gt_missing_diacritic_is_error: bool = False):
        if len(voweled_sentences) != len(ground_truth_sentences):
            raise RuntimeError("Sentences length mismatch")
        Total_WER_count, Morph_WER_count, Total_DER_count, Morph_DER_count = 0, 0, 0, 0
        total_word_count, total_letter_count, total_not_voweled_words_count = 0, 0, 0
        scored = 0

        for i in range(len(voweled_sentences)):
            try:
                wc, lc, nvw, w_twer, w_mwer, w_tder, w_mder = cls.caculate_error_on_single_sentence(
                    voweled_sentences[i].strip(), ground_truth_sentences[i].strip(), gt_missing_diacritic_is_error
                )
                total_word_count += wc
                total_letter_count += lc
                total_not_voweled_words_count += nvw
                Total_WER_count += w_twer
                Morph_WER_count += w_mwer
                Total_DER_count += w_tder
                Morph_DER_count += w_mder
                scored += 1
            except RuntimeError:
                continue

        Total_WER = Total_WER_count / max(total_word_count, 1) * 100
        Morph_WER = Morph_WER_count / max(total_word_count, 1) * 100
        Total_DER = Total_DER_count / max(total_letter_count, 1) * 100
        Morph_DER = Morph_DER_count / max(total_letter_count, 1) * 100
        NVW = total_not_voweled_words_count / max(total_word_count, 1) * 100

        return Total_WER, Morph_WER, Total_DER, Morph_DER, NVW
