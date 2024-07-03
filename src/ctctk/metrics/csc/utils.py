#!/usr/bin/env python
# encoding: utf-8
# Adapted from: https://github.com/nghuyong/cscd-ime/blob/master/evaluation/util.py

import unicodedata


def compute_p_r_f1(true_predict, all_predict, all_error):
    """
    @param true_predict:
    @param all_predict:
    @param all_error:
    @return:
    """
    if all_predict == 0:
        p = 0.0
    else:
        p = round(true_predict / all_predict * 100, 3)
    if all_error == 0:
        r = 0.0
    else:
        r = round(true_predict / all_error * 100, 3)
    f1 = round(2 * p * r / (p + r + 1e-10), 3)
    return {'p': p, 'r': r, 'f1': f1}


def write_report(output_file, metric, output_errors):
    """
    generate report
    @param output_file:
    @param metric:
    @param output_errors:
    @return:
    """
    with open(output_file, 'wt', encoding='utf-8') as f:
        f.write('overview:\n')
        for key in sorted(metric.keys()):
            f.write(f'{key}:\t{metric[key]}\n')
        f.write('\nbad cases:\n')
        for output_error in output_errors:
            f.write("\n".join(output_error))
            f.write("\n\n")


def input_check_and_process(src_sentences, tgt_sentences, pred_sentences):
    """
    check the input is valid
    @param src_sentences:
    @param tgt_sentences:
    @param pred_sentences:
    @return:
    """
    assert len(src_sentences) == len(tgt_sentences) == len(pred_sentences)
    src_char_list = [list(s) for s in src_sentences]
    tgt_char_list = [list(s) for s in tgt_sentences]
    pred_char_list = [list(s) for s in pred_sentences]
    return src_char_list, tgt_char_list, pred_char_list

def to_halfwidth_char(char):
    if u"\uff01" <= char <= u"\uff5e":
        return chr(ord(char) - 0xfee0)
    else:
        return char

def to_halfwidth(sentence):
    return "".join([to_halfwidth_char(char) for char in sentence])

def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)